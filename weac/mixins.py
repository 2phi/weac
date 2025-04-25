"""Mixins for the elastic analysis of layered snow slabs."""
# pylint: disable=invalid-name,too-many-locals,too-many-arguments,too-many-lines

# Standard library imports
from functools import partial

# Third party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

# Module imports
from weac.tools import calc_vertical_bc_center_of_gravity, tensile_strength_slab


class FieldQuantitiesMixin:
    """
    Mixin for computing field quantities in snow slab analysis.

    This mixin provides methods for calculating various mechanical quantities
    from the solution vector, including:
    - Displacements (u, v, w)
    - Rotations (psi_x, psi_y, psi_z)
    - Stresses (normal and shear)
    - Strains
    - Energy release rates

    The methods in this mixin are used to post-process the solution vector
    obtained from the governing equations of the snow slab.
    """

    # pylint: disable=no-self-use
    def psix(self, Z, unit="rad"):
        """
        Calculate the torsion of a section plane around the x-axis.

        This method computes the rotation angle psi_x around the x-axis at each
        point along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        unit : {'deg', 'degrees', 'rad', 'radians'}, optional
            Desired output unit. Default is radians.

        Returns
        -------
        psi : float
            Cross-section rotation psi_x (in specified unit) of the slab.
        """
        if unit in ["deg", "degree", "degrees"]:
            psix = np.rad2deg(Z[6, :])
        elif unit in ["rad", "radian", "radians"]:
            psix = Z[6, :]
        return psix

    def dpsix_dx(self, Z):
        """
        Calculate the first derivative of the section torsion around the x-axis.

        This method computes the rate of change of the rotation angle psi_x
        along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.

        Returns
        -------
        float
            First derivative psi_x' of the midplane rotation (radians/mm)
            of the slab.
        """
        return Z[7, :]

    def psiy(self, Z, unit="rad"):
        """
        Calculate the midplane rotation around the y-axis.

        This method computes the rotation angle psi_y around the y-axis at each
        point along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        unit : {'deg', 'degrees', 'rad', 'radians'}, optional
            Desired output unit. Default is radians.

        Returns
        -------
        psi : float
            Cross-section rotation psi_y (in specified unit) of the slab.
        """
        if unit in ["deg", "degree", "degrees"]:
            psiy = np.rad2deg(Z[8, :])
        elif unit in ["rad", "radian", "radians"]:
            psiy = Z[8, :]
        return psiy

    def dpsiy_dx(self, Z):
        """
        Calculate the first derivative of the midplane rotation around the y-axis.

        This method computes the rate of change of the rotation angle psi_y
        along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            First derivative psi_y' of the midplane rotation (radians/mm)
            of the slab.
        """
        return Z[9, :]

    def psiz(self, Z, unit="rad"):
        """
        Calculate the midplane rotation around the z-axis.

        This method computes the rotation angle psi_z around the z-axis at each
        point along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        unit : {'deg', 'degrees', 'rad', 'radians'}, optional
            Desired output unit. Default is radians.

        Returns
        -------
        psi : float
            Cross-section rotation psi_z (in specified unit) of the slab.
        """
        if unit in ["deg", "degree", "degrees"]:
            psiz = np.rad2deg(Z[10, :])
        elif unit in ["rad", "radian", "radians"]:
            psiz = Z[10, :]
        return psiz

    def dpsiz_dx(self, Z):
        """
        Calculate the first derivative of the midplane rotation around the z-axis.

        This method computes the rate of change of the rotation angle psi_z
        along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            First derivative psi_z' of the midplane rotation (radians/mm)
            of the slab.
        """
        return Z[11, :]

    def w(self, Z, y0=0, unit="mm"):
        """
        Calculate the centerline deflection in the z-direction.

        This method computes the vertical displacement w at each point along
        the slab length, taking into account the offset from the centerline.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Deflection w (in specified unit) of the slab.
        """
        convert = {
            "m": 1e-3,  # meters
            "cm": 1e-1,  # centimeters
            "mm": 1,  # millimeters
            "um": 1e3,  # micrometers
        }
        return convert[unit] * (Z[4, :] + y0 * self.psix(Z))

    def dw_dx(self, Z, y0=0):
        """
        Calculate the first derivative of the centerline deflection.

        This method computes the rate of change of the vertical displacement w
        along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.

        Returns
        -------
        float
            First derivative w' of the deflection of the slab.
        """
        return Z[5, :] + y0 * self.dpsix_dx(Z)

    def u(self, Z, z0=0, y0=0, unit="mm"):
        """
        Calculate the axial displacement u = u0 + z0 psiy - y0 psix.

        This method computes the horizontal displacement u at each point along
        the slab length, taking into account the offsets from the centerline.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Horizontal displacement u (in specified unit) of the slab.
        """
        convert = {
            "m": 1e-3,  # meters
            "cm": 1e-1,  # centimeters
            "mm": 1,  # millimeters
            "um": 1e3,  # micrometers
        }
        u = convert[unit] * (Z[0, :] + z0 * self.psiy(Z) - y0 * self.psiz(Z))
        return u

    def du_dx(self, Z, z0=0, y0=0):
        """
        Calculate the first derivative of the axial displacement.

        This method computes the rate of change of the horizontal displacement u
        along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float
            Offset from the centerline in the z-direction. Defailt is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.

        Returns
        -------
        float
            First derivative du_dx of the horizontal displacement of the slab.
        """
        return Z[1, :] + z0 * self.dpsiy_dx(Z) - y0 * self.dpsiz_dx(Z)

    def v(self, Z, z0=0, unit="mm"):
        """
        Calculate the centerline deflection in the y-direction.

        This method computes the out-of-plane displacement v at each point along
        the slab length, taking into account the offset from the centerline.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Deflection v (in specified unit) of the slab.
        """
        convert = {
            "m": 1e-3,  # meters
            "cm": 1e-1,  # centimeters
            "mm": 1,  # millimeters
            "um": 1e3,  # micrometers
        }
        return convert[unit] * (Z[2, :] - z0 * self.psix(Z))

    def dv_dx(self, Z, z0=0):
        """
        Calculate the first derivative of the centerline deflection in the y-direction.

        This method computes the rate of change of the out-of-plane displacement v
        along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.

        Returns
        -------
        float
            First derivative v' of the out-of-plane displacement of the slab.
        """
        return Z[3, :] - z0 * self.dpsix_dx(Z)

    def theta_uc(self, Z):
        """
        Calculate the axial constant displacement in the weak layer.

        This method computes the rotation angle at the center of the slab
        based on the solution vector.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Axial displacement at the center of the weak layer (mm).
        """
        return Z[12, :]

    def dtheta_uc_dx(self, Z):
        """
        Calculate the first derivative of the constant dispalcemnet in at the center of the weak layer.

        This method computes the rate of change of the constant dispalcemnet in at the center of the weak layer along its length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.


        Returns
        -------
        float
            First derivative of the axial displacement at the center of the weak layer (mm/mm).
        """
        return Z[13, :]

    def theta_ul(self, Z):
        """
        Calculate the linear amplitude of axial cosine shaped displacements in the weak layer.

        This method computes the linear amplitude of axial cosine shaped displacements in the weak layer.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Linear amplitude theta_uk (mm) of the slab.
        """
        return Z[14, :]

    def dtheta_ul_dx(self, Z):
        """
        Calculate the first derivative of the linear amplitude of axial cosine shaped displacements in the weak layer.

        This method computes the rate of change of the linear amplitude of axial cosine shaped displacements in the weak layer.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            First derivative of the linear amplitude of axial cosine shaped displacements in the weak layer (mm/mm).
        """
        return Z[15, :]

    def theta_vc(self, Z):
        """
        Calculate the constant amplitude of out-of-plane cosine shaped displacements in the weak layer theta_vc.

        This method computes the constant amplitude of out-of-plane cosine shaped displacements in the weak layer theta_vc
        based on the solution vector.

        Arguments
        ---------
        Z : ndarray
            Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            constant amplitude of out-of-plane cosine shaped displacements in the weak layer theta_vc (mm).
        """
        return Z[16, :]

    def dtheta_vc_dx(self, Z):
        """
        Calculate the first derivative of the constant amplitude of out-of-plane cosine shaped displacements in the weak layer.

        This method computes the rate of change the constant amplitude of out-of-plane cosine shaped displacements in the weak layer along its length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            First derivative the constant amplitude of out-of-plane cosine shaped displacements in the weak layer (mm/mm).
        """
        return Z[17, :]

    def theta_vl(self, Z):
        """
        Calculate the linear amplitude of out-of-plane cosine shaped displacements in the weak layer theta_vk .

        This method computes linear amplitude of out-of-plane cosine shaped displacements in the weak layer theta_vk
        based on the solution vector.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.


        Returns
        -------
        float
            Linear amplitude theta_vk (mm).
        """
        return Z[18, :]

    def dtheta_vl_dx(self, Z):
        """
        Calculate the first derivative of the linear amplitude of out-of-plane cosine shaped displacements in the weak layer.

        This method computes the rate of change of the linear amplitude of out-of-plane cosine shaped displacements in the weak layer along its length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            First derivative of the linear amplitude of out-of-plane cosine shaped displacements in the weak layer (mm/mm).
        """
        return Z[19, :]

    def theta_wc(self, Z):
        """
        Calculate the constant amplitude of vertical cosine shaped displacements in the weak layer.

        This method computes the constant amplitude of vertical cosine shaped displacements in the weak layer
        based on the solution vector.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            constant amplitude of vertical cosine shaped displacements in the weak layer (mm).
        """
        return Z[20, :]

    def dtheta_wc_dx(self, Z):
        """
        Calculate the constant amplitude of vertical cosine shaped displacements in the weak layer.

        This method computes the rate of change of the constant amplitude of vertical cosine shaped displacements in the weak layer along its length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            First derivative of the constant amplitude of vertical cosine shaped displacements in the weak layer (mm/mm).
        """
        return Z[21, :]

    def theta_wl(self, Z):
        """
        Calculate the linear amplitude of vertical cosine shaped displacements in the weak layer.

        This method computes the linear amplitude of vertical cosine shaped displacements in the weak layer
        based on the solution vector.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Linear amplitude of vertical cosine shaped displacements in the weak layer (mm).
        """
        return Z[22, :]

    def dtheta_wl_dx(self, Z):
        """
        Calculate the first derivative of the linear amplitude of vertical cosine shaped displacements in the weak layer.

        This method computes the rate of change of the linear amplitude of vertical cosine shaped displacements in the weak layer along its length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            First derivative ofthe linear amplitude of vertical cosine shaped displacements in the weak layer (mm/mm).
        """
        return Z[23, :]

    def uweak(self, Z, z0, y0=0, unit="mm"):
        """
        Calculate the displacement in the weak layer in the x-direction.

        This method computes the horizontal displacement in the weak layer
        at each point along the slab length, taking into account the offsets
        from the centerline.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float
            Offset from the centerline in the z-direction.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Axial displacement in the weak layer (in specified unit).
        """
        convert = {
            "m": 1e-3,  # meters
            "cm": 1e-1,  # centimeters
            "mm": 1,  # millimeters
            "um": 1e3,  # micrometers
        }
        # Unpack geometrical properties of layerd

        h = self.h
        t = self.t
        Pi = np.pi
        b = self.b

        return convert[unit] * (
            self.u(Z, h / 2, y0) * (1 - (z0 - h / 2) / t)
            + np.cos(Pi * (2 * z0 - h - t) / (2 * t))
            * (self.theta_uc(Z) + (2 * y0 / b) * self.theta_ul(Z))
        )

    def vweak(self, Z, z0, y0=0, unit="mm"):
        """
        Calculate the displacement in the weak layer in the y-direction.

        This method computes the horizontal displacement in the weak layer
        at each point along the slab length, taking into account the offsets
        from the centerline.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float
            Offset from the centerline in the z-direction.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Horizontal displacement in the weak layer (in specified unit).
        """
        convert = {
            "m": 1e-3,  # meters
            "cm": 1e-1,  # centimeters
            "mm": 1,  # millimeters
            "um": 1e3,  # micrometers
        }
        # Unpack geometrical properties of layerd

        h = self.h
        t = self.t
        Pi = np.pi
        b = self.b

        # Return vweak
        return convert[unit] * (
            self.v(Z, h / 2, y0) * (1 - (z0 - h / 2) / t)
            + np.cos(Pi * (2 * z0 - h - t) / (2 * t))
            * (self.theta_vc(Z) + (2 * y0 / b) * self.theta_vl(Z))
        )

    def wweak(self, Z, z0, y0=0, unit="mm"):
        """
        Calculate the displacement in the weak layer in the z-direction.

        This method computes the vertical displacement in the weak layer
        at each point along the slab length, taking into account the offsets
        from the centerline.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float
            Offset from the centerline in the z-direction.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Vertical displacement in the weak layer (in specified unit).
        """
        convert = {
            "m": 1e-3,  # meters
            "cm": 1e-1,  # centimeters
            "mm": 1,  # millimeters
            "um": 1e3,  # micrometers
        }
        # Unpack geometrical properties of layerd
        h = self.h
        t = self.t
        Pi = np.pi
        b = self.b

    def Nxx(self, Z, bed=False):
        """
        Calculate the normal force in the x-direction.

        This method computes the normal force in the x-direction at each point
        along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        bed : bool, optional
            Whether to include the bed effect. Default is False.

        Returns
        -------
        float
            Normal force in the x-direction (N).
        """
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h

        Nxx = self.A11 * b * Z[1, :] + self.B11 * b * Z[9, :]
        if bed:
            Nxx = Nxx + (
                Ew
                * (
                    3 * b * Pi * nuw * Z[4, :]
                    - 12 * t * nuw * self.theta_vl(Z)
                    - 12 * b * nuw * self.theta_wc(Z)
                    + b
                    * t
                    * (-1 + nuw)
                    * (
                        2 * Pi * Z[1, :]
                        + 6 * self.dtheta_uc_dx(Z)
                        + h * Pi * self.dpsiy_dx(Z)
                    )
                )
            ) / (6 * Pi * (-1 + nuw + 2 * nuw**2))
        return Nxx

    def Myy(self, Z, bed=False):
        """
        Calculate the bending moment around the y-axis.

        This method computes the bending moment around the y-axis at each point
        along the slab length.

        Arguments
        ---------
         Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        bed : bool, optional
            Whether to include the bed effect. Default is False.

        Returns
        -------
        float
            Bending moment around the y-axis (N·mm).
        """
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h
        Myy = self.B11 * b * Z[1, :] + self.D11 * b * Z[9, :]
        if bed:
            Myy = Myy + (
                h
                * Ew
                * (
                    3 * b * Pi * nuw * Z[4, :]
                    - 12 * t * nuw * self.theta_vl(Z)
                    - 12 * b * nuw * self.theta_wc(Z)
                    + b
                    * t
                    * (-1 + nuw)
                    * (
                        2 * Pi * Z[1, :]
                        + 6 * self.dtheta_uc_dx(Z)
                        + h * Pi * self.dpsiy_dx(Z)
                    )
                )
            ) / (12 * Pi * (-1 + nuw + 2 * nuw**2))
        return Myy

    def Mxx(self, Z, bed=False):
        """
        Calculate the bending moment around the x-axis.

        This method computes the bending moment around the x-axis at each point
        along the slab length.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        bed : bool, optional
            Whether to include the bed effect. Default is False.

        Returns
        -------
        float
            Torsion moment around the x-axis (N·mm).
        """
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h
        Mxx = (
            self.kA55 * np.longdouble(b**3 / 12) * Z[7, :]
            + self.kB55 * b * (Z[10, :] - Z[3, :])
            + self.kD55 * b * Z[7, :]
        )
        if bed:
            Mxx = Mxx + np.longdouble(
                (
                    -1
                    / 144
                    * (
                        b**2
                        * Ew
                        * (
                            -24 * t * self.theta_ul(Z)
                            + 3 * b * Pi * (-2 + t) * self.psiz(Z)
                            + 12 * (-2 + t) * t * self.dtheta_wl_dx(Z)
                            + b * Pi * (-3 + t) * t * self.dpsix_dx(Z)
                        )
                    )
                    / (Pi * t * (1 + nuw))
                    - (
                        h
                        * t
                        * Ew
                        * (
                            12 * self.theta_ul(Z)
                            + b
                            * (
                                -2 * Pi * self.psiz(Z)
                                + 2 * Pi * Z[3, :]
                                + 6 * self.dtheta_vc_dx(Z)
                                - h * Pi * self.dpsix_dx(Z)
                            )
                        )
                    )
                    / (24 * Pi * (1 + nuw))
                )
            )
        return Mxx

    def Mzz(self, Z, bed=False):
        """
        Calculate the bending moment around the z-axis.

        This method computes the bending moment around the z-axis at each point
        along the slab length.
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        bed : bool, optional
            Whether to include the bed effect. Default is False.
        Returns
        -------
        float
            Bending moment around the z-axis (N mm).
        """
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h
        Mzz = self.A11 * np.longdouble(b**3) / 12 * Z[11, :]
        if bed:
            Mzz = Mzz - np.longdouble(
                b**2
                * Ew
                * (
                    -24 * nuw * self.theta_wl(Z)
                    + 3 * b * Pi * nuw * self.psix(Z)
                    + 2
                    * t
                    * (-1 + nuw)
                    * (6 * self.dtheta_ul_dx(Z) - b * Pi * self.dpsiz_dx(Z))
                )
            ) / (72 * Pi * (-1 + nuw + 2 * nuw**2))
        return Mzz

    def Vyy(self, Z, bed=False):
        """
        Calculate the shear force in y-direction.

        This method computes the shear force in y-direction at each point
        along the slab length.
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        bed : bool, optional
            Whether to include the bed effect. Default is False.

        Returns
        -------
        float
            Out-of-plane shear force (N)).
        """
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h
        Vyy = self.kA55 * b * (Z[3, :] - Z[10, :]) - self.kB55 * b * Z[7, :]
        if bed:
            Vyy = Vyy + np.longdouble(
                t
                * Ew
                * (
                    12 * self.theta_ul(Z)
                    + b
                    * (
                        -2 * Pi * self.psiz(Z)
                        + 2 * Pi * Z[3, :]
                        + 6 * self.dtheta_vc_dx(Z)
                        - h * Pi * self.dpsix_dx(Z)
                    )
                )
            ) / (12 * Pi * (1 + nuw))
        return Vyy

    def Vzz(self, Z, bed=False):
        """
        Calculate the shear force in the z-direction.

        This method computes the shear force in y-direction at each point
        along the slab length.

        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        bed : bool, optional
            Whether to include the bed effect. Default is False.


        Returns
        -------
        float
            Vertical shear force V (N) in the slab.
        """
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h

        Vzz = self.kA55 * b * (Z[5, :] + Z[8, :])
        if bed:
            Vzz = Vzz + np.longdouble(
                b
                * Ew
                * (
                    -6 * Pi * Z[0, :]
                    + 24 * self.theta_uc(Z)
                    - 3 * h * Pi * self.psiy(Z)
                    + 4 * Pi * t * Z[5, :]
                    + 12 * t * self.dtheta_wc_dx(Z)
                )
            ) / (24 * Pi * (1 + nuw))
        return Vzz

    def NxxWL(self, Z):
        """
        Calculate the normal force in the weak layer in the x-direction.

        This method computes the normal force in the weak layer in the x-direction
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Normal force in the weak layer in the x-direction (N).
        """
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h

        return (
            Ew
            * (
                4 * b * nuw * Z[4, :]
                - 2 * Pi * t * nuw * self.theta_vl(Z)
                + b
                * t
                * (-1 + nuw)
                * (2 * Z[1, :] + Pi * self.dtheta_uc_dx(Z) + h * self.dpsiy_dx(Z))
            )
        ) / (2 * Pi * (1 + nuw) * (-1 + 2 * nuw))

    def VyyWL(self, Z):
        """
        Calculate the shear force in the weak layer in the y-direction.

        This method computes the shear force in the weak layer in the y-direction
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Shear force in the weak layer in the y-direction (N).
        """
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h

        return (
            t
            * Ew
            * (
                2 * Pi * self.theta_ul(Z)
                + b
                * (
                    -2 * self.psiz(Z)
                    + 2 * Z[3, :]
                    + Pi * self.dtheta_vc_dx(Z)
                    - h * self.dpsix_dx(Z)
                )
            )
        ) / (4 * Pi * (1 + nuw))

    def VzzWL(self, Z):
        """
        Calculate the shear force in the weak layer in the z-direction.

        This method computes the shear force in the weak layer in the z-direction
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Shear force in the weak layer in the z-direction (N).
        """
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h

        return (
            b
            * Ew
            * (
                -4 * Z[0, :]
                - 2 * h * self.psiy(Z)
                + 2 * t * Z[5, :]
                + Pi * t * self.dtheta_wc_dx(Z)
            )
        ) / (4 * Pi * (1 + nuw))

    def MxxWL(self, Z):
        """
        Calculate the associated force with theta_ul in the weak layer.

        This method computes the associated force with theta_ul in the weak layer
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Associated force with theta_ul in the weak layer (N).
        """
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h

        return (
            b
            * Ew
            * (
                2 * b * self.psiz(Z)
                + Pi * t * self.dtheta_wl_dx(Z)
                + b * t * self.dpsix_dx(Z)
            )
        ) / (12 * Pi * (1 + nuw))

    def MyyWL(self, Z):
        """
        Calculate the associated force with theta_vl in the weak layer.

        This method computes the associated force with theta_vl in the weak layer
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Associated force with theta_vl in the weak layer (N).
        """
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h

        return (b * t * Ew * self.dtheta_vl_dx(Z)) / (12 * (1 + nuw))

    def MzzWL(self, Z):
        """
        Calculate the associated force with theta_wl in the weak layer.

        This method computes the associated force with theta_wl in the weak layer
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.

        Returns
        -------
        float
            Associated force with theta_wl in the weak layer (N).
        """
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        t = self.t
        Pi = np.pi
        b = self.b
        h = self.h

        return (
            b
            * Ew
            * (
                2 * b * nuw * self.psix(Z)
                + t * (-1 + nuw) * (Pi * self.dtheta_ul_dx(Z) - b * self.dpsiz_dx(Z))
            )
        ) / (6 * Pi * (1 + nuw) * (-1 + 2 * nuw))

    def sigzz(self, Z, z0=0, y0=0, unit="MPa"):
        """
        Calculate the normal stress in the weak layer in the z-direction.

        This method computes the normal stress in the weak layer in the z-direction
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'Pa', 'kPa', 'MPa', 'GPa'}, optional
            Desired output unit. Default is MPa.

        Returns
        -------
        float
            Normal stress in the weak layer in the z-direction (in specified unit).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        convert = {
            "Pa": 1e6,  # pascals
            "kPa": 1e3,  # kilopascals
            "MPa": 1,  # megapascals
            "GPa": 1e-3,  # gigapascals
        }
        # Adjust z-coordinate for weak layer
        z0 = z0 + self.h / 2 + self.t / 2
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        b = self.b
        h = self.h
        t = self.t
        Pi = np.pi
        return (
            convert[unit]
            * (
                2
                * Ew
                * nuw
                * np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                * self.theta_vl(Z)
            )
            / (b * (1 - 2 * nuw) * (1 + nuw))
            + (
                Ew
                * (1 - nuw)
                * (
                    -(
                        (
                            Pi
                            * np.sin((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                            * (self.theta_wc(Z) + (2 * y0 * self.theta_wl(Z)) / b)
                        )
                        / t
                    )
                    - (Z[4, :] + y0 * self.psix(Z)) / t
                )
            )
            / ((1 - 2 * nuw) * (1 + nuw))
            + (
                Ew
                * nuw
                * (
                    np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                    * (self.dtheta_uc_dx(Z) + (2 * y0 * self.dtheta_ul_dx(Z)) / b)
                    + (1 - (-1 / 2 * h + z0) / t)
                    * (Z[1, :] + (h * self.dpsiy_dx(Z)) / 2 - y0 * self.dpsiz_dx(Z))
                )
            )
            / ((1 - 2 * nuw) * (1 + nuw))
        )

    def tauxz(self, Z, z0=0, y0=0, unit="MPa"):
        """
        Calculate the shear stress in the weak layer in the x-z plane.

        This method computes the shear stress in the weak layer in the x-z plane
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'Pa', 'kPa', 'MPa', 'GPa'}, optional
            Desired output unit. Default is MPa.

        Returns
        -------
        float
            Shear stress in the weak layer in the x-z plane (in specified unit).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        convert = {
            "Pa": 1e6,  # pascals
            "kPa": 1e3,  # kilopascals
            "MPa": 1,  # megapascals
            "GPa": 1e-3,  # gigapascals
        }
        # Adjust z-coordinate for weak layer
        z0 = z0 + self.h / 2 + self.t / 2
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        b = self.b
        h = self.h
        t = self.t
        Pi = np.pi
        return (
            convert[unit]
            * (
                Ew
                * (
                    -(
                        (
                            Pi
                            * np.sin((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                            * (self.theta_uc(Z) + (2 * y0 * self.theta_ul(Z)) / b)
                        )
                        / t
                    )
                    - (Z[0, :] + (h * self.psiy(Z)) / 2 - y0 * self.psiz(Z)) / t
                    + np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                    * (self.dtheta_wc_dx(Z) + (2 * y0 * self.dtheta_wl_dx(Z)) / b)
                    + (1 - (-1 / 2 * h + z0) / t) * (Z[5, :] + y0 * self.dpsix_dx(Z))
                )
            )
            / (2 * (1 + nuw))
        )

    def tauxy(self, Z, z0=0, y0=0, unit="MPa"):
        """
        Calculate the shear stress in the weak layer in the x-y plane.

        This method computes the shear stress in the weak layer in the x-y plane
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'Pa', 'kPa', 'MPa', 'GPa'}, optional
            Desired output unit. Default is MPa.

        Returns
        -------
        float
            Shear stress in the weak layer in the x-y plane (in specified unit).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        convert = {
            "Pa": 1e6,  # pascals
            "kPa": 1e3,  # kilopascals
            "MPa": 1,  # megapascals
            "GPa": 1e-3,  # gigapascals
        }
        # Adjust z-coordinate for weak layer
        z0 = z0 + self.h / 2 + self.t / 2
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        b = self.b
        h = self.h
        t = self.t
        Pi = np.pi
        return (
            convert[unit]
            * (
                Ew
                * (
                    (2 * np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t) * self.theta_ul(Z))
                    / b
                    - (1 - (-1 / 2 * h + z0) / t) * self.psiz(Z)
                    + np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                    * (self.dtheta_vc_dx(Z) + (2 * y0 * self.dtheta_vl_dx(Z)) / b)
                    + (1 - (-1 / 2 * h + z0) / t)
                    * (Z[3, :] - (h * self.dpsix_dx(Z)) / 2)
                )
            )
            / (2 * (1 + nuw))
        )

    def tauyz(self, Z, z0=0, y0=0, unit="MPa"):
        """
        Calculate the shear stress in the weak layer in the x-y plane.

        This method computes the shear stress in the weak layer in the x-y plane
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.
        unit : {'Pa', 'kPa', 'MPa', 'GPa'}, optional
            Desired output unit. Default is MPa.

        Returns
        -------
        float
            Shear stress in the weak layer in the y-z plane (in specified unit).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        convert = {
            "Pa": 1e6,  # pascals
            "kPa": 1e3,  # kilopascals
            "MPa": 1,  # megapascals
            "GPa": 1e-3,  # gigapascals
        }
        # Adjust z-coordinate for weak layer
        z0 = z0 + self.h / 2 + self.t / 2
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        b = self.b
        h = self.h
        t = self.t
        Pi = np.pi
        return (
            convert[unit]
            * (
                Ew
                * (
                    -(
                        (
                            Pi
                            * np.sin((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                            * (self.theta_vc(Z) + (2 * y0 * self.theta_vl(Z)) / b)
                        )
                        / t
                    )
                    + (
                        2
                        * np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                        * self.theta_wl(Z)
                    )
                    / b
                    + (1 - (-1 / 2 * h + z0) / t) * self.psix(Z)
                    - (Z[2, :] - (h * self.psix(Z)) / 2) / t
                )
            )
            / (2 * (1 + nuw))
        )

    def epszz(self, Z, z0=0, y0=0):
        """
        Calculate the normal strain in the weak layer in the z-direction.

        This method computes the normal strain in the weak layer in the z-direction
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.

        Returns
        -------
        float
            Normal strain in the weak layer in the z-direction.
        """
        # Unpack weak layer material properties
        b = self.b
        h = self.h
        t = self.t
        Pi = np.pi
        # Adjust z-coordinate for weak layer
        z0 = z0 + h / 2 + t / 2

        return (
            -(
                (
                    Pi
                    * np.sin((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                    * (self.theta_wc(Z) + (2 * y0 * self.theta_wl(Z)) / b)
                )
                / t
            )
            - (Z[4, :] + y0 * self.psix(Z)) / t
        )

    def gammaxz(self, Z, z0=0, y0=0):
        """
        Calculate the shear strain in the weak layer in the x-z plane.

        This method computes the shear strain in the weak layer in the x-z plane
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.

        Returns
        -------
        float
            Shear strain in the weak layer in the x-z plane.
        """
        # Unpack weak layer material properties
        b = self.b
        h = self.h
        t = self.t
        Pi = np.pi

        return (
            -(
                (
                    Pi
                    * np.sin((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
                    * (self.theta_uc(Z) + (2 * y0 * self.theta_ul(Z)) / b)
                )
                / t
            )
            - (Z[0, :] + (h * self.psiy(Z)) / 2 - y0 * self.psiz(Z)) / t
            + np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
            * (self.dtheta_wc_dx(Z) + (2 * y0 * self.dtheta_wl_dx(Z)) / b)
            + (1 - (-1 / 2 * h + z0) / t) * (Z[5, :] + y0 * self.dpsix_dx(Z))
        )

    def gammaxy(self, Z, z0=0, y0=0):
        """
        Calculate the shear strain in the weak layer in the x-y plane.

        This method computes the shear strain in the weak layer in the x-y plane
        at each point along the slab length, taking into account the weak layer
        properties and geometry.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        z0 : float, optional
            Offset from the centerline in the z-direction. Default is 0.
        y0 : float, optional
            Offset from the centerline in the y-direction. Default is 0.

        Returns
        -------
        float
            Shear strain in the weak layer in the x-y plane.
        """
        # Unpack weak layer material properties
        b = self.b
        h = self.h
        t = self.t
        Pi = np.pi

        return (
            (2 * np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t) * self.theta_ul(Z)) / b
            - (1 - (-1 / 2 * h + z0) / t) * self.psiz(Z)
            + np.cos((Pi * (-1 / 2 * h - t / 2 + z0)) / t)
            * (self.dtheta_vc_dx(Z) + (2 * y0 * self.dtheta_vl_dx(Z)) / b)
            + (1 - (-1 / 2 * h + z0) / t) * (Z[3, :] - (h * self.dpsix_dx(Z)) / 2)
        )

    def Gi(self, Ztip, Zback, unit="kJ/m^2"):
        """
        Calculate the energy release rate (ERR) for mode I (opening mode) fracture.

        This method computes the energy release rate for mode I fracture at the crack tip,
        taking into account the stress and displacement fields in the weak layer.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector at the crack tip [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        Zback : ndarray
            Solution vector at the end of the supported segment [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        unit : {'J/m^2', 'kJ/m^2'}, optional
            Desired output unit. Default is kJ/m^2.

        Returns
        -------
        float
            Energy release rate for mode I fracture (in specified unit).
        """
        # Convert energy release rate from J/m^2 to specified unit
        convert = {
            "J/m^2": 1,  # joules per square meter
            "kJ/m^2": 1e-3,  # kilojoules per square meter
        }
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        rhow = self.weak["rho"]
        phi = self.phi
        theta = self.theta
        b = self.b
        t = self.t
        Pi = np.pi
        h = self.h
        g = 9810

        convert = {
            "J/m^2": 1e3,  # joule per square meter
            "kJ/m^2": 1,  # kilojoule per square meter
            "N/mm": 1,  # newton per millimeter
        }
        return (
            convert[unit]
            * 1
            / b
            * (
                -1
                / 2
                * (
                    b
                    * g
                    * rhow
                    * t
                    * np.cos(phi)
                    * np.cos(theta)
                    * (
                        -Pi * Zback[4, :]
                        + Pi * Ztip[4, :]
                        - 4 * self.theta_wc(Zback)
                        + 4 * self.theta_wc(Ztip)
                    )
                )
                / Pi
                + (
                    Ew
                    * (
                        24 * b * Pi * (-1 + nuw) * Ztip[4, :] ** 2
                        + 6
                        * t
                        * nuw
                        * Ztip[4, :]
                        * (
                            16 * self.theta_vl(Ztip)
                            + b
                            * (
                                2 * Pi * Ztip[1, :]
                                + 8 * self.dtheta_uc_dx(Ztip)
                                + h * Pi * self.dpsiy_dx(Ztip)
                            )
                        )
                        + b
                        * (
                            12 * Pi**3 * (-1 + nuw) * self.theta_wc(Ztip) ** 2
                            + 4 * Pi**3 * (-1 + nuw) * self.theta_wl(Ztip) ** 2
                            - 24
                            * t
                            * nuw
                            * self.theta_wc(Ztip)
                            * (2 * Ztip[1, :] + h * self.dpsiy_dx(Ztip))
                            + 8 * b * t * nuw * self.theta_wl(Ztip) * self.dpsiz_dx(Ztip)
                            + b
                            * self.psix(Ztip)
                            * (
                                2 * b * Pi * (-1 + nuw) * self.psix(Ztip)
                                + t
                                * nuw
                                * (
                                    8 * self.dtheta_ul_dx(Ztip)
                                    - b * Pi * self.dpsiz_dx(Ztip)
                                )
                            )
                        )
                    )
                )
                / (48 * Pi * t * (1 + nuw) * (-1 + 2 * nuw))
            )
        )

    def Gii(self, Ztip, Zback, unit="kJ/m^2"):
        """
        Calculate the energy release rate (ERR) for mode II (sliding mode) fracture.

        This method computes the energy release rate for mode II fracture at the crack tip,
        taking into account the stress and displacement fields in the weak layer.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector at the crack tip [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        Zback : ndarray
            Solution vector at the crack back [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        unit : {'J/m^2', 'kJ/m^2'}, optional
            Desired output unit. Default is kJ/m^2.

        Returns
        -------
        float
            Energy release rate for mode II fracture (in specified unit).
        """
        # Convert energy release rate from J/m^2 to specified unit
        convert = {
            "J/m^2": 1,  # joules per square meter
            "kJ/m^2": 1e-3,  # kilojoules per square meter
        }
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        rhow = self.weak["rho"]
        t = self.t
        Pi = np.pi
        h = self.h
        b = self.b
        phi = self.phi
        g = 9810
        theta = self.theta

        convert = {
            "J/m^2": 1e3,  # joule per square meter
            "kJ/m^2": 1,  # kilojoule per square meter
            "N/mm": 1,  # newton per millimeter
        }
        return (
            convert[unit]
            * 1
            / b
            * (
                (
                    b
                    * g
                    * rhow
                    * t
                    * np.sin(phi)
                    * (
                        -2 * Pi * Zback[0, :]
                        + 2 * Pi * Ztip[0, :]
                        - 8 * self.theta_uc(Zback)
                        + 8 * self.theta_uc(Ztip)
                        - h * Pi * self.psiy(Zback)
                        + h * Pi * self.psiy(Ztip)
                    )
                )
                / (4 * Pi)
                + (
                    (
                        b
                        * Ew
                        * (
                            36 * Pi * Ztip[0, :] ** 2
                            + 36
                            * Ztip[0, :]
                            * (
                                h * Pi * self.psiy(Ztip)
                                - t * (Pi * Ztip[5, :] + 4 * self.dtheta_wc_dx(Ztip))
                            )
                            + 3
                            * (
                                6 * Pi**3 * self.theta_uc(Ztip) ** 2
                                + 2 * Pi**3 * self.theta_ul(Ztip) ** 2
                                + 3 * h**2 * Pi * self.psiy(Ztip) ** 2
                                + b**2 * Pi * self.psiz(Ztip) ** 2
                                + 48 * t * self.theta_uc(Ztip) * Ztip[5, :]
                                - 6 * h * Pi * t * self.psiy(Ztip) * Ztip[5, :]
                                + 4 * Pi * t**2 * Ztip[5, :] ** 2
                                - 24 * h * t * self.psiy(Ztip) * self.dtheta_wc_dx(Ztip)
                                + 24 * t**2 * Ztip[5, :] * self.dtheta_wc_dx(Ztip)
                                + 6 * Pi * t**2 * self.dtheta_wc_dx(Ztip) ** 2
                                + 8 * b * t * self.psiz(Ztip) * self.dtheta_wl_dx(Ztip)
                                + 2 * Pi * t**2 * self.dtheta_wl_dx(Ztip) ** 2
                            )
                            + 3
                            * b
                            * t
                            * (
                                8 * self.theta_ul(Ztip)
                                + b * Pi * self.psiz(Ztip)
                                + 4 * t * self.dtheta_wl_dx(Ztip)
                            )
                            * self.dpsix_dx(Ztip)
                            + b**2 * Pi * t**2 * self.dpsix_dx(Ztip) ** 2
                        )
                    )
                    / (144 * Pi * t * (1 + nuw))
                )
            )
        )

    def Giii(self, Ztip, Zback, unit="kJ/m^2"):
        """
        Calculate the energy release rate (ERR) for mode III (tearing mode) fracture.

        This method computes the energy release rate for mode III fracture at the crack tip,
        taking into account the stress and displacement fields in the weak layer.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector at the crack tip [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        Zback : ndarray
            Solution vector at the crack back [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        unit : {'J/m^2', 'kJ/m^2'}, optional
            Desired output unit. Default is kJ/m^2.

        Returns
        -------
        float
            Energy release rate for mode III fracture (in specified unit).
        """
        # Convert energy release rate from J/m^2 to specified unit
        convert = {
            "J/m^2": 1,  # joules per square meter
            "kJ/m^2": 1e-3,  # kilojoules per square meter
        }
        # Unpack weak layer material properties
        Ew = self.weak["E"]
        nuw = self.weak["nu"]
        rhow = self.weak["rho"]
        t = self.t
        Pi = np.pi
        h = self.h
        b = self.b
        phi = self.phi
        g = 9810
        theta = self.theta

        convert = {
            "J/m^2": 1e3,  # joule per square meter
            "kJ/m^2": 1,  # kilojoule per square meter
            "N/mm": 1,  # newton per millimeter
        }

        return (
            convert[unit]
            * 1
            / b
            * (
                -1
                / 4
                * (
                    b
                    * g
                    * rhow
                    * t
                    * np.sin(theta)
                    * (
                        -2 * Pi * Zback[2, :]
                        + 2 * Pi * Ztip[2, :]
                        - 8 * self.theta_vc(Zback)
                        + 8 * self.theta_vc(Ztip)
                        + h * Pi * self.psix(Zback)
                        - h * Pi * self.psix(Ztip)
                    )
                )
                / Pi
                + (
                    (
                        Ew
                        * (
                            12 * b**2 * Pi * Ztip[2, :] ** 2
                            + 2
                            * b**2
                            * Pi**3
                            * (3 * self.theta_vc(Ztip) ** 2 + self.theta_vl(Ztip) ** 2)
                            + 24 * Pi * t**2 * self.theta_wl(Ztip) ** 2
                            + 48
                            * b
                            * t
                            * (b * self.theta_vc(Ztip) + (h + t) * self.theta_wl(Ztip))
                            * self.psix(Ztip)
                            + b**2
                            * Pi
                            * (3 * h**2 + 6 * h * t + 4 * t**2)
                            * self.psix(Ztip) ** 2
                            - 12
                            * b
                            * Ztip[2, :]
                            * (
                                8 * t * self.theta_wl(Ztip)
                                + b * Pi * (h + t) * self.psix(Ztip)
                            )
                        )
                    )
                    / (48 * b * Pi * t * (1 + nuw))
                )
            )
        )

    def G(self, Ztip, Zback, unit="kJ/m^2"):
        """
        Calculate the total energy release rate (ERR) for mixed-mode fracture.

        This method computes the total energy release rate at the crack tip by summing
        the contributions from all three fracture modes (I, II, and III).

        Arguments
        ---------
        Ztip : ndarray
            Solution vector at the crack tip [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        Zback : ndarray
            Solution vector at the crack back [u(x) u'(x) v(x) v'(x) w(x) w'(x) psi_x(x) psi_x'(x) psi_y(x) psi_y'(x) psi_z(x) psi_z'(x)
              theta_uc(x) theta_uc'(x) theta_ul(x) theta_ul'(x) theta_vc(x) theta_vc'(x) theta_vl(x) theta_vl'(x)
              theta_wc(x) theta_wc'(x) theta_wl(x) theta_wl'(x)]^T.
        unit : {'J/m^2', 'kJ/m^2'}, optional
            Desired output unit. Default is kJ/m^2.

        Returns
        -------
        float
            Total energy release rate (in specified unit).
        """
        # Calculate individual mode contributions
        Gi = self.Gi(Ztip, Zback, unit)
        Gii = self.Gii(Ztip, Zback, unit)
        Giii = self.Giii(Ztip, Zback, unit)

        # Sum all contributions
        return Gi + Gii + Giii

    def int1(self, x, z0, z1):
        """
        Calculate the mode I crack opening integrand at specified integration points.

        This method computes the integrand for the mode I crack opening integral,
        which is used to calculate the energy release rate for mode I fracture.

        Arguments
        ---------
        x : float or ndarray
            X-coordinate(s) where the integrand is to be evaluated (in mm).
        z0 : callable
            Function that returns the solution vector for the uncracked configuration.
            The function should take x as input and return the solution vector.
        z1 : callable
            Function that returns the solution vector for the cracked configuration.
            The function should take x as input and return the solution vector.

        Returns
        -------
        float or ndarray
            Integrand of the mode I crack opening integral at the specified points.
        """
        # Calculate the integrand by multiplying stress and displacement
        return -self.sig(z0(x)) * self.w(z1(x))

    def int2(self, x, z0, z1):
        """
        Calculate the mode II crack opening integrand at specified integration points.

        This method computes the integrand for the mode II crack opening integral,
        which is used to calculate the energy release rate for mode II fracture.

        Arguments
        ---------
        x : float or ndarray
            X-coordinate(s) where the integrand is to be evaluated (in mm).
        z0 : callable
            Function that returns the solution vector for the uncracked configuration.
            The function should take x as input and return the solution vector.
        z1 : callable
            Function that returns the solution vector for the cracked configuration.
            The function should take x as input and return the solution vector.

        Returns
        -------
        float or ndarray
            Integrand of the mode II crack opening integral at the specified points.
        """
        # Calculate the integrand by multiplying shear stress, shear strain, and thickness
        return -self.tauxz(z0(x)) * self.u(z1(x), z0=self.h / 2)

    def int3(self, x, z0, z1):
        """
        Calculate the mode III crack opening integrand at specified integration points.

        This method computes the integrand for the mode III crack opening integral,
        which is used to calculate the energy release rate for mode III fracture.

        Arguments
        ---------
        x : float or ndarray
            X-coordinate(s) where the integrand is to be evaluated (in mm).
        z0 : callable
            Function that returns the solution vector for the uncracked configuration.
            The function should take x as input and return the solution vector.
        z1 : callable
            Function that returns the solution vector for the cracked configuration.
            The function should take x as input and return the solution vector.

        Returns
        -------
        float or ndarray
            Integrand of the mode III crack opening integral at the specified points.
        """
        # Calculate the integrand by multiplying shear stress, shear strain, and thickness
        return -self.tauyz(z0(x)) * self.v(z1(x), z0=self.h / 2)


class SolutionMixin:
    """
    A mixin class that provides methods for solving the beam equations and handling boundary conditions.

    This class contains methods for:
    - Calculating mode shapes and natural frequencies
    - Reducing system stiffness
    - Applying boundary conditions
    - Solving the system of equations
    - Assembling and solving the complete system
    """

    def mode_td(self, l=0):
        """
        Calculate the mode shapes and natural frequencies for a given length.

        This method computes the mode shapes and natural frequencies of the beam
        for a specified length parameter.

        Arguments
        ---------
        l : float, optional
            Length parameter for the calculation. Default is 0.

        Returns
        -------
        tuple
            A tuple containing:
            - mode_shapes : ndarray
                Array of mode shapes
            - frequencies : ndarray
                Array of natural frequencies
        """
        # Calculate mode shapes and frequencies
        # Implementation details...
        pass

    def reduce_stiffness(self, l=0, mode="A"):
        """
        Reduce the system stiffness matrix for a given length and mode.

        This method reduces the stiffness matrix of the system based on the
        specified length parameter and mode.

        Arguments
        ---------
        l : float, optional
            Length parameter for the calculation. Default is 0.
        mode : {'A', 'B', 'C'}, optional
            Mode of reduction. Default is 'A'.

        Returns
        -------
        ndarray
            Reduced stiffness matrix.
        """
        # Reduce system stiffness
        # Implementation details...
        pass

    def bc(self, z, l=0, k=False, pos="mid"):
        """
        Apply boundary conditions to the system.

        This method applies the appropriate boundary conditions to the system
        based on the specified parameters.

        Arguments
        ---------
        z : ndarray
            Solution vector to which boundary conditions are applied.
        l : float, optional
            Length parameter. Default is 0.
        k : bool, optional
            Flag for supported (True) and unsupported(False) segments. Default is False.
        pos : {'mid', 'left', 'right'}, optional
            Position for boundary condition application. Default is 'mid'.

        Returns
        -------
        ndarray
            Vector of boundary conditions at position x.
        """
        # Check mode for free end
        mode = "A"  # self.mode_td(l=l)
        # Get spring stiffness reduction factor
        kf = self.reduce_stiffness(l=l, mode=mode)
        # Get spring stiffness for collapsed weak-layer
        kR = self.calc_rot_spring(collapse=True)

        # Set boundary conditions for PST-systems
        if self.system in ["pst-", "-pst", "skier-finite"]:
            factor = -1 if pos in ["left", "l"] else 1
            if not k:
                if mode in ["A"]:
                    # Free end
                    # Factor for correct sign

                    bc = factor * np.array(
                        [
                            self.Nxx(z, bed=k),
                            self.Vyy(z, bed=k),
                            self.Vzz(z, bed=k),
                            self.Mxx(z, bed=k),
                            self.Myy(z, bed=k),
                            self.Mzz(z, bed=k),
                        ]
                    )
                elif mode in ["B", "C"] and pos in ["r", "right"]:
                    # Touchdown right
                    bc = np.array(
                        [
                            self.Nxx(z, bed=k),
                            self.Vyy(z, bed=k),
                            self.Vzz(z, bed=k),
                            self.Myy(z, bed=k) + kf * kR * self.psi(z),
                            self.w(z, z0=0, y0=0),
                        ]
                    )
                elif mode in ["B", "C"] and pos in ["l", "left"]:
                    # Touchdown left
                    bc = np.array(
                        [
                            self.Nxx(z, bed=k),
                            self.Myy(z, bed=k) - kf * kR * self.psi(z),
                            self.w(z, z0=0, y0=0),
                        ]
                    )
            else:
                # Free end
                bc = factor * np.array(
                    [
                        self.Nxx(z, bed=k),
                        self.Vyy(z, bed=k),
                        self.Vzz(z, bed=k),
                        self.Mxx(z, bed=k),
                        self.Myy(z, bed=k),
                        self.Mzz(z, bed=k),
                        self.NxxWL(z),
                        self.VyyWL(z),
                        self.VzzWL(z),
                        self.MxxWL(z),
                        self.MyyWL(z),
                        self.MzzWL(z),
                    ]
                )
        # Set boundary conditions for SKIER-systems
        elif self.system in ["skier", "skiers"]:
            # Infinite end (vanishing complementary solution)
            if not k:
                bc = np.array(
                    [
                        self.u(z, z0=0, y0=0),
                        self.w(z, y0=0),
                        self.v(z, z0=0),
                        self.psix(z),
                        self.psiy(z),
                        self.psiz(z),
                    ]
                )
            else:
                bc = np.array(
                    [
                        self.u(z, z0=0, y0=0),
                        self.v(z, z0=0),
                        self.w(z, y0=0),
                        self.psix(z),
                        self.psiy(z),
                        self.psiz(z),
                        self.theta_uc(z),
                        self.theta_ul(z),
                        self.theta_vc(z),
                        self.theta_vl(z),
                        self.theta_wc(z),
                        self.theta_wl(z),
                    ]
                )
        else:
            raise ValueError(
                f"Boundary conditions not defined forsystem of type {self.system}."
            )

        return bc

    def eqs(self, zl, zr, l=0, k=False, pos="mid"):
        """
        Set up the system of equations for the beam.

        This method sets up the system of equations that describe the beam's
        behavior based on the specified parameters.

        Arguments
        ---------
        zl : ndarray
            Left boundary solution vector. Size is (12x1) or (24x1) depending on foundation.
        zr : ndarray
            Right boundary solution vector. Size is (12x1) or (24x1) depending on foundation.
        l : float, optional
            Length parameter. Default is 0.
        k : bool, optional
            Flag for support. Default is False.
        pos : {'mid', 'left', 'right'}, optional
            Position for equation setup. Default is 'mid'.

        Returns
        -------
        eqsSlab : ndarray
            Vector (of length 18) of boundary conditions for the slab (6) and
            transmission conditions for the slab (12) for boundary segements
            or vector of transmission conditions for the slab (of length 12+12)
            for center segments.
        eqsWeak: ndarray
            Vector (of length 18 ) of boundary conditions for the weak layer (6) and
            transmission conditions for the weak layer (12) for boundary segements
            or vector of transmission conditions for the weak layer (of length 24)
            for center segments.
        """
        # Handle unsupported segments (k=False)
        if not k:
            # Left boundary segment
            if pos in ("l", "left"):
                # Set up equations for left boundary
                # First 6 elements are boundary conditions from bc method
                # Next 12 elements are transmission conditions at right end
                eqsSlab = np.array(
                    [
                        self.bc(zl, l, k, pos)[0],  # Left boundary condition
                        self.bc(zl, l, k, pos)[1],  # Left boundary condition
                        self.bc(zl, l, k, pos)[2],  # Left boundary condition
                        self.bc(zl, l, k, pos)[3],
                        self.bc(zl, l, k, pos)[4],
                        self.bc(zl, l, k, pos)[5],
                        self.u(zr, z0=0, y0=0),  # Displacement at right end
                        self.v(zr, z0=0),  # Vertical displacement
                        self.w(zr, y0=0),  # Lateral displacement
                        self.psix(zr),  # Rotation about x
                        self.psiy(zr),  # Rotation about y
                        self.psiz(zr),  # Rotation about z
                        self.Nxx(zr, bed=k),  # Normal force
                        self.Vyy(zr, bed=k),  # Shear force y
                        self.Vzz(zr, bed=k),  # Shear force z
                        self.Mxx(zr, bed=k),  # Bending moment x
                        self.Myy(zr, bed=k),  # Bending moment y
                        self.Mzz(zr, bed=k),
                    ]
                )  # Bending moment z

            # Middle segment
            elif pos in ("m", "mid"):
                # Set up equations for middle segment
                # First 12 elements are transmission conditions at left end
                # Next 12 elements are transmission conditions at right end
                eqsSlab = np.array(
                    [
                        -self.u(zl, z0=0, y0=0),  # Negative displacement at left end
                        -self.v(zl, z0=0),  # Negative vertical displacement
                        -self.w(zl, y0=0),  # Negative lateral displacement
                        -self.psix(zl),  # Negative rotation about x
                        -self.psiy(zl),  # Negative rotation about y
                        -self.psiz(zl),  # Negative rotation about z
                        -self.Nxx(zl, bed=k),  # Negative normal force
                        -self.Vyy(zl, bed=k),  # Negative shear force y
                        -self.Vzz(zl, bed=k),  # Negative shear force z
                        -self.Mxx(zl, bed=k),  # Negative bending moment x
                        -self.Myy(zl, bed=k),  # Negative bending moment y
                        -self.Mzz(zl, bed=k),  # Negative bending moment z
                        self.u(zr, z0=0, y0=0),  # Displacement at right end
                        self.v(zr, z0=0),  # Vertical displacement
                        self.w(zr, y0=0),  # Lateral displacement
                        self.psix(zr),  # Rotation about x
                        self.psiy(zr),  # Rotation about y
                        self.psiz(zr),  # Rotation about z
                        self.Nxx(zr, bed=k),  # Normal force
                        self.Vyy(zr, bed=k),  # Shear force y
                        self.Vzz(zr, bed=k),  # Shear force z
                        self.Mxx(zr, bed=k),  # Bending moment x
                        self.Myy(zr, bed=k),  # Bending moment y
                        self.Mzz(zr, bed=k),
                    ]
                )  # Bending moment z

            # Right boundary segment
            elif pos in ("r", "right"):
                # Set up equations for right boundary
                # First 12 elements are transmission conditions at left end
                # Last 6 elements are boundary conditions from bc method
                eqsSlab = np.array(
                    [
                        -self.u(zl, z0=0, y0=0),  # Negative displacement at left end
                        -self.v(zl, z0=0),  # Negative vertical displacement
                        -self.w(zl, y0=0),  # Negative lateral displacement
                        -self.psix(zl),  # Negative rotation about x
                        -self.psiy(zl),  # Negative rotation about y
                        -self.psiz(zl),  # Negative rotation about z
                        -self.Nxx(zl, bed=k),  # Negative normal force
                        -self.Vyy(zl, bed=k),  # Negative shear force y
                        -self.Vzz(zl, bed=k),  # Negative shear force z
                        -self.Mxx(zl, bed=k),  # Negative bending moment x
                        -self.Myy(zl, bed=k),  # Negative bending moment y
                        -self.Mzz(zl, bed=k),  # Negative bending moment z
                        self.bc(zr, l, k, pos)[0],  # Right boundary condition
                        self.bc(zr, l, k, pos)[1],  # Right boundary condition
                        self.bc(zr, l, k, pos)[2],  # Right boundary condition
                        self.bc(zr, l, k, pos)[3],  # Right boundary condition
                        self.bc(zr, l, k, pos)[4],  # Right boundary condition
                        self.bc(zr, l, k, pos)[5],
                    ]
                )  # Right boundary condition

            else:
                raise ValueError(
                    (
                        f"Invalid position argument {pos} given. "
                        "Valid segment positions are l, m, and r, "
                        "or left, mid and right."
                    )
                )

            # For unsupported segments, weak layer equations are zero
            eqsWeak = np.zeros((12, eqsSlab.shape[1]))

        # Handle supported segments (k=True)
        else:
            # Left boundary segment
            if pos in ("l", "left"):
                # Set up equations for left boundary with support
                # First 6 elements are boundary conditions from bc method
                # Next 12 elements are transmission conditions at right end
                eqsSlab = np.array(
                    [
                        self.bc(zl, l, k, pos)[0],  # Left boundary condition
                        self.bc(zl, l, k, pos)[1],  # Left boundary condition
                        self.bc(zl, l, k, pos)[2],  # Left boundary condition
                        self.bc(zl, l, k, pos)[3],  # Left boundary condition
                        self.bc(zl, l, k, pos)[4],  # Left boundary condition
                        self.bc(zl, l, k, pos)[5],  # Left boundary condition
                        self.u(zr, z0=0, y0=0),  # Displacement at right end
                        self.v(zr, z0=0),  # Vertical displacement
                        self.w(zr, y0=0),  # Lateral displacement
                        self.psix(zr),  # Rotation about x
                        self.psiy(zr),  # Rotation about y
                        self.psiz(zr),  # Rotation about z
                        self.Nxx(zr, bed=k),  # Normal force
                        self.Vyy(zr, bed=k),  # Shear force y
                        self.Vzz(zr, bed=k),  # Shear force z
                        self.Mxx(zr, bed=k),  # Bending moment x
                        self.Myy(zr, bed=k),  # Bending moment y
                        self.Mzz(zr, bed=k),
                    ]
                )  # Bending moment z

                # Set up weak layer equations for left boundary
                # First 6 elements are boundary conditions from bc method
                # Next 12 elements are transmission conditions at right end
                eqsWeak = np.array(
                    [
                        self.bc(zl, l, k, pos)[
                            6
                        ],  # Left boundary condition in weak layer
                        self.bc(zl, l, k, pos)[
                            7
                        ],  # Left boundary condition in weak layer
                        self.bc(zl, l, k, pos)[
                            8
                        ],  # Left boundary condition in weak layer
                        self.bc(zl, l, k, pos)[
                            9
                        ],  # Left boundary condition in weak layer
                        self.bc(zl, l, k, pos)[
                            10
                        ],  # Left boundary condition in weak layer
                        self.bc(zl, l, k, pos)[
                            11
                        ],  # Left boundary condition in weak layer
                        self.theta_uc(zr),  # Rotation in weak layer
                        self.theta_ul(zr),  # Rotation in weak layer
                        self.theta_vc(zr),  # Rotation in weak layer
                        self.theta_vl(zr),  # Rotation in weak layer
                        self.theta_wc(zr),  # Rotation in weak layer
                        self.theta_wl(zr),  # Rotation in weak layer
                        self.NxxWL(zr),  # Normal force in weak layer
                        self.VyyWL(zr),  # Shear force in weak layer
                        self.VzzWL(zr),  # Shear force in weak layer
                        self.MxxWL(zr),  # Bending moment in weak layer
                        self.MyyWL(zr),  # Bending moment in weak layer
                        self.MzzWL(zr),
                    ]
                )  # Bending moment in weak layer

            # Middle segment
            elif pos in ("m", "mid"):
                # Set up equations for middle segment with support
                # First 12 elements are transmission conditions at left end
                # Next 12 elements are transmission conditions at right end
                eqsSlab = np.array(
                    [
                        -self.u(zl, z0=0, y0=0),  # Negative displacement at left end
                        -self.v(zl, z0=0),  # Negative vertical displacement
                        -self.w(zl, y0=0),  # Negative lateral displacement
                        -self.psix(zl),  # Negative rotation about x
                        -self.psiy(zl),  # Negative rotation about y
                        -self.psiz(zl),  # Negative rotation about z
                        -self.Nxx(zl, bed=k),  # Negative normal force
                        -self.Vyy(zl, bed=k),  # Negative shear force y
                        -self.Vzz(zl, bed=k),  # Negative shear force z
                        -self.Mxx(zl, bed=k),  # Negative bending moment x
                        -self.Myy(zl, bed=k),  # Negative bending moment y
                        -self.Mzz(zl, bed=k),  # Negative bending moment z
                        self.u(zr, z0=0, y0=0),  # Displacement at right end
                        self.v(zr, z0=0),  # Vertical displacement
                        self.w(zr, y0=0),  # Lateral displacement
                        self.psix(zr),  # Rotation about x
                        self.psiy(zr),  # Rotation about y
                        self.psiz(zr),  # Rotation about z
                        self.Nxx(zr, bed=k),  # Normal force
                        self.Vyy(zr, bed=k),  # Shear force y
                        self.Vzz(zr, bed=k),  # Shear force z
                        self.Mxx(zr, bed=k),  # Bending moment x
                        self.Myy(zr, bed=k),  # Bending moment y
                        self.Mzz(zr, bed=k),
                    ]
                )  # Bending moment z

                # Set up weak layer equations for middle segment
                # First 12 elements are transmission conditions at left end
                # Next 12 elements are transmission conditions at right end
                eqsWeak = np.array(
                    [
                        -self.theta_uc(zl),  # Negative rotation in weak layer
                        -self.theta_ul(zl),  # Negative rotation in weak layer
                        -self.theta_vc(zl),  # Negative rotation in weak layer
                        -self.theta_vl(zl),  # Negative rotation in weak layer
                        -self.theta_wc(zl),  # Negative rotation in weak layer
                        -self.theta_wl(zl),  # Negative rotation in weak layer
                        -self.NxxWL(zl),  # Negative normal force in weak layer
                        -self.VyyWL(zl),  # Negative shear force in weak layer
                        -self.VzzWL(zl),  # Negative shear force in weak layer
                        -self.MxxWL(zl),  # Negative bending moment in weak layer
                        -self.MyyWL(zl),  # Negative bending moment in weak layer
                        -self.MzzWL(zl),  # Negative bending moment in weak layer
                        self.theta_uc(zr),  # Rotation in weak layer
                        self.theta_ul(zr),  # Rotation in weak layer
                        self.theta_vc(zr),  # Rotation in weak layer
                        self.theta_vl(zr),  # Rotation in weak layer
                        self.theta_wc(zr),  # Rotation in weak layer
                        self.theta_wl(zr),  # Rotation in weak layer
                        self.NxxWL(zr),  # Normal force in weak layer
                        self.VyyWL(zr),  # Shear force in weak layer
                        self.VzzWL(zr),  # Shear force in weak layer
                        self.MxxWL(zr),  # Bending moment in weak layer
                        self.MyyWL(zr),  # Bending moment in weak layer
                        self.MzzWL(zr),
                    ]
                )  # Bending moment in weak layer

            # Right boundary segment
            elif pos in ("r", "right"):
                # Set up equations for right boundary with support
                # First 12 elements are transmission conditions at left end
                # Last 6 elements are boundary conditions from bc method
                eqsSlab = np.array(
                    [
                        -self.u(zl, z0=0, y0=0),  # Negative displacement at left end
                        -self.v(zl, z0=0),  # Negative vertical displacement
                        -self.w(zl, y0=0),  # Negative lateral displacement
                        -self.psix(zl),  # Negative rotation about x
                        -self.psiy(zl),  # Negative rotation about y
                        -self.psiz(zl),  # Negative rotation about z
                        -self.Nxx(zl, bed=k),  # Negative normal force
                        -self.Vyy(zl, bed=k),  # Negative shear force y
                        -self.Vzz(zl, bed=k),  # Negative shear force z
                        -self.Mxx(zl, bed=k),  # Negative bending moment x
                        -self.Myy(zl, bed=k),  # Negative bending moment y
                        -self.Mzz(zl, bed=k),  # Negative bending moment z
                        self.bc(zr, l, k, pos)[0],  # Right boundary condition
                        self.bc(zr, l, k, pos)[1],  # Right boundary condition
                        self.bc(zr, l, k, pos)[2],  # Right boundary condition
                        self.bc(zr, l, k, pos)[3],  # Right boundary condition
                        self.bc(zr, l, k, pos)[4],  # Right boundary condition
                        self.bc(zr, l, k, pos)[5],
                    ]
                )  # Right boundary condition

                # Set up weak layer equations for right boundary
                # First 12 elements are transmission conditions at left end
                # Last 6 elements are boundary conditions from bc method
                eqsWeak = np.array(
                    [
                        -self.theta_uc(zl),  # Negative rotation in weak layer
                        -self.theta_ul(zl),  # Negative rotation in weak layer
                        -self.theta_vc(zl),  # Negative rotation in weak layer
                        -self.theta_vl(zl),  # Negative rotation in weak layer
                        -self.theta_wc(zl),  # Negative rotation in weak layer
                        -self.theta_wl(zl),  # Negative rotation in weak layer
                        -self.NxxWL(zl),  # Negative normal force in weak layer
                        -self.VyyWL(zl),  # Negative shear force in weak layer
                        -self.VzzWL(zl),  # Negative shear force in weak layer
                        -self.MxxWL(zl),  # Negative bending moment in weak layer
                        -self.MyyWL(zl),  # Negative bending moment in weak layer
                        -self.MzzWL(zl),  # Negative bending moment in weak layer
                        self.bc(zr, l, k, pos)[
                            6
                        ],  # Right boundary condition in weak layer
                        self.bc(zr, l, k, pos)[
                            7
                        ],  # Right boundary condition in weak layer
                        self.bc(zr, l, k, pos)[
                            8
                        ],  # Right boundary condition in weak layer
                        self.bc(zr, l, k, pos)[
                            9
                        ],  # Right boundary condition in weak layer
                        self.bc(zr, l, k, pos)[
                            10
                        ],  # Right boundary condition in weak layer
                        self.bc(zr, l, k, pos)[11],
                    ]
                )  # Right boundary condition in weak layer

            else:
                raise ValueError(
                    (
                        f"Invalid position argument {pos} given. "
                        "Valid segment positions are l, m, and r, "
                        "or left, mid and right."
                    )
                )

        return eqsSlab, eqsWeak

    def calc_segments(
        self,
        tdi=False,
        li=False,
        ki=False,
        k0=False,
        wi=False,
        fi=False,
        L=1e4,
        a=0,
        m=0,
        phi=0,
        theta=0,
        **kwargs,
    ):
        """
        Assemble lists defining the segments for different beam systems.

        This function creates segment definitions for various beam systems including:
        - 'skiers': Multiple segments with skier weights
        - 'pst-': PST with crack at left end
        - '-pst': PST with crack at right end
        - 'skier': Single skier load in center
        - 'skier-finite': Finite length version of skier system

        Each segment is defined by:
        - Length (li): Length of each segment in mm
        - Skier weight (mi): Weight of skier at segment boundaries in kg
        - Foundation in cracked state (ki): Boolean indicating if segment has foundation
        - Foundation in uncracked state (k0): Boolean indicating if segment has foundation
        - Additional weights (wi): Boolean indicating if segment has additional weights

        Arguments
        ---------
        tdi : bool, optional
            Touchdown indicator (not currently used)
        li : sequence, optional
            List of segment lengths in mm. Used for system 'skiers'
        mi : sequence, optional
            List of skier weights in kg at segment boundaries. Used for system 'skiers'
        ki : sequence, optional
            List of booleans indicating foundation in cracked state for each segment
        k0 : sequence, optional
            List of booleans indicating foundation in uncracked state for each segment
        wi : sequence, optional
            List of booleans indicating additional weights for each segment
        fi : sequence, optional
            List of lists with the transmission and boundary loadvectors in N and Nmm
        L : float, optional
            Total length of model (mm). Used for systems 'pst-', '-pst',
            and 'skier'.
        a : float, optional
            Crack length (mm).  Used for systems 'pst-', '-pst', and
            'skier'.
        b : float, optional
            Loaded length (mm).  Used for systems 'modeIII-pst-loaded'.
        phi : float, optional
            Inclination (degree).
        theta : float, optional
            Tilt (degree).
        m : float, optional
            Weight of skier in kg in the axial center of the model. Used for system 'skier'

        Returns
        -------
        segments : dict
            Dictionary containing three configurations:
            - 'nocrack': Segment definitions for uncracked state
            - 'crack': Segment definitions for cracked state
            - 'both': Combined segment definitions for both states
            Each configuration contains lists for lengths (li), skier weights (mi),
            foundation states (ki/k0), and additional weights (wi)
        """
        # Store unused arguments
        _ = kwargs

        # Determine unbedded segment length based on touchdown mode
        mode = "A"  # Deactivate touchdown mode
        if mode in ["A", "B"]:
            lU = a  # Unbedded length equals crack length
        if mode in ["C"]:
            lU = self.lS  # Use stored segment length

        # Assemble segment definitions based on system type
        if self.system == "skiers":
            # Convert input lists to numpy arrays for skiers system
            li = np.array(li)  # Segment lengths
            ki = np.array(ki)  # Foundation in cracked state
            k0 = np.array(k0)  # Foundation in uncracked state
            wi = np.array(wi)  # Additional loads
            fi = np.array(fi)  # Loadvectors at segment boundaries

        elif self.system == "pst-":
            # PST with crack at left end
            li = np.array([L - a, lU])  # [supported length, unsupported length]
            ki = np.array([True, False])  # Foundation only in supported segment
            k0 = np.array([True, True])  # Foundation in both segments
            wi = np.array([False, False])  # Additional surface loads on both segments
            fi = np.array(
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            )  # Loadvectors at segment boundaries

        elif self.system == "-pst":
            # PST with crack at right end
            li = np.array([lU, L - a])  # [unsupported length, supported length]
            ki = np.array([False, True])  # Foundation only in supported segment
            k0 = np.array([True, True])  # Foundation in both segments
            wi = np.array([False, False])  # Additional surface load on both segments
            fi = np.array(
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            )  # Loadvectors at segment boundaries

        elif self.system in ["skier", "skier-finite"]:
            # Single skier load in center
            Fx, Fy, Fz = self.get_skier_load(m, phi, theta)
            lb = (L - a) / 2  # Half supported length
            lf = a / 2  # Half free length
            li = np.array(
                [lb, lf, lf, lb]
            )  # [left supported, left free, right free, right supported]
            ki = np.array([True, False, False, True])  # Foundation in supported segments
            k0 = np.array([True, True, True, True])  # Foundation in all segments
            wi = np.array(
                [True, True, True, True]
            )  # Additional surface loads on all segments
            fi = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [Fx, Fy, Fz, Fy * self.h / 2, -Fx * self.h / 2, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )  # Loadvectors at segment boundaries

        else:
            raise ValueError(f"System {self.system} is not implemented.")

        # Create dictionary with segment definitions for different states
        segments = {
            "nocrack": {"li": li, "fi": fi, "ki": k0, "wi": wi},  # Uncracked state
            "crack": {"li": li, "fi": fi, "ki": ki, "wi": wi},  # Cracked state
            "both": {"li": li, "fi": fi, "ki": ki, "k0": k0, "wi": wi},  # Both states
        }
        return segments

    def assemble_and_solve(self, phi, theta, li, ki, wi, fi):
        """
        Assemble and solve the system of equations from boundary and transmission conditions.

        Assemble LHS for slabs transition and boundary conditions from supported and unsupported segments in the form
        [       ]   [  zh1    0     0   ...   0     0     0   ][   ]   [      ]   [       ]  left
        [       ]   [  zh1   zh2    0   ...   0     0     0   ][   ]   [      ]   [       ]  mid
        [       ]   [   0    zh2   zh3  ...   0     0     0   ][   ]   [      ]   [       ]  mid
        [z0Slab ]   [  ...   ...   ...  ...  ...   ...   ...  ][   ]   [  zp  ]   [  rhs  ]  mid
        [       ]   [   0     0     0   ...  zhL   zhM    0   ][   ]   [      ]   [       ]  mid
        [       ]   [   0     0     0   ...   0    zhM   zhN  ][   ]   [      ]   [       ]  mid
        [       ]   [   0     0     0   ...   0     0    zhN  ][   ]   [      ]   [       ]  right
        [       ] = [ zh1wl   0     0   ...   0     0     0   ][ C ] + [      ] = [       ]  left
        [       ]   [ zh1wl zh2wl   0   ...   0     0     0   ][   ]   [      ]   [       ]  mid
        [       ]   [   0   zh2wl zh3wl ...   0     0     0   ][   ]   [      ]   [       ]  mid
        [z0Weak ]   [  ...   ...   ...  ...  ...   ...   ...  ][   ]   [ zpwl ]   [ rhswl ]  mid
        [       ]   [   0     0     0   ... zhLwl zhMwl   0   ][   ]   [      ]   [       ]  mid
        [       ]   [   0     0     0   ...   0   zhMwl zhNwl ][   ]   [      ]   [       ]  mid
        [       ]   [   0     0     0   ...   0     0   zhNwl ][   ]   [      ]   [       ]  right


        Where:
        - [zhi], [zhiwl]: Entries in the homogeneous solution matrices for the slab and the weak layer
        - [zp], [zpwl]: Particular solution vectors for the slab and the weak layer
        - [C]: Unknown constants to be solved for
        - [rhs], [rhswl]: Right-hand side vectors for the slab and the weak layer

        Arguments
        ---------
        phi : float
            Inclination (degrees).
        li : ndarray
            List of lengths of segements (mm).
        fi : ndarray
            List of loadvectors at segment boundaries (N) and (Nmm).
        ki : ndarray
            Array of booleans indicating foundation for each segment
        wi : ndarray
            Array of booleans indicating additional surface loads for each segment


        Returns
        -------
        C : ndarray
            Matrix of shpae (24 x N)  for a system of N segements. For unsupported segments, the last 12 entries in the column are NaN.
        """
        # --- ERROR CHECKING ----------------------------------------------------

        # Verify at least one segment has foundation
        if not any(ki):
            raise ValueError("Provide at least one bedded segment.")

        # Verify consistent number of segments and transitions
        if len(li) != len(ki) or len(li) + 1 != len(fi):
            raise ValueError(
                "Make sure len(li)=N, len(ki)=N, and "
                "len(fi)=N+1 for a system of N segments."
            )

        # Check boundary conditions for infinite systems
        if self.system not in ["pst-", "-pst"]:
            # Boundary segments must be on foundation for infinite BCs
            if not all([ki[0], ki[-1]]):
                raise ValueError(
                    "Provide bedded boundary segments in "
                    "order to account for infinite extensions."
                )
            # Verify boundary segments are long enough
            if li[0] < 5e3 or li[-1] < 5e3:
                print(
                    (
                        "WARNING: Boundary segments are short. Make sure "
                        "the complementary solution has decayed to the "
                        "boundaries."
                    )
                )

        # --- SYSTEM SETUP ---------------------------------------------------

        # Calculate system dimensions
        nSBedded = ki.sum()  # Number of bedded segments
        nSFree = len(ki) - ki.sum()  # Number of free segments
        nS = nSBedded + nSFree  # Total number of segments
        nDOFfree = 12  # DOF per free segment
        nDOFbedded = 24  # DOF per bedded segment

        # Add dummy segment if only one segment provided
        if nS == 1:
            li.append(0)
            ki.append(True)
            fi.append([0, 0, 0, 0, 0, 0])
            wi.append(True)
            nS = 2

        # Initialize position vector (l=left, m=middle, r=right)
        pi = np.full(nS, "m")
        pi[0], pi[-1] = "l", "r"

        # Initialize matrices for the slab
        zh0Slab = np.zeros(
            [
                nSBedded * nDOFfree + nSFree * nDOFfree,
                nSBedded * nDOFbedded + nSFree * nDOFfree,
            ],
            dtype=np.double,
        )
        zp0Slab = np.zeros([nSBedded * nDOFfree + nSFree * nDOFfree, 1], dtype=np.double)
        rhsSlab = np.zeros([nSBedded * nDOFfree + nSFree * nDOFfree, 1], dtype=np.double)

        # Initialize matrices for weak layer
        zh0Weak = np.zeros(
            [
                nSBedded * (nDOFbedded - nDOFfree),
                nSBedded * nDOFbedded + nSFree * nDOFfree,
            ],
            dtype=np.double,
        )
        zp0Weak = np.zeros([nSBedded * (nDOFbedded - nDOFfree), 1], dtype=np.double)
        rhsWeak = np.zeros([nSBedded * (nDOFbedded - nDOFfree), 1], dtype=np.double)

        # --- ASSEMBLE EQUATIONS ---------------------------------------------
        globalStartSlab = 0  # Global counter for slab matrix position
        globalStartWeak = 0  # Global counter for weak layer matrix position
        kprev = False  # Initalize bedding storage
        # Loop through segments to assemble equations
        for i in range(nS):
            # Get segment properties
            l, k, pos, w = li[i], ki[i], pi[i], wi[i]

            # Transmission conditions at left and right segment ends
            zhiSlab, zhiWeak = self.eqs(
                zl=np.around(self.zh(x=0, l=l, bed=k), 16),
                zr=np.around(self.zh(x=l, l=l, bed=k), 16),
                l=l,
                k=k,
                pos=pos,
            )
            zpiSlab, zpiWeak = self.eqs(
                zl=np.around(self.zp(x=0, phi=phi, theta=theta, bed=k, load=w), 16),
                zr=np.around(self.zp(x=l, phi=phi, theta=theta, bed=k, load=w), 16),
                l=l,
                k=k,
                pos=pos,
            )

            # Assemble slab equations
            nConst = 24 if k else 12  # Constants per segment
            startSlab = 0 if i == 0 else 6
            stopSlab = 12 if i == nS - 1 else 18
            zh0Slab[
                (12 * i - startSlab) : (12 * i + stopSlab),
                globalStartSlab : globalStartSlab + nConst,
            ] = zhiSlab
            zp0Slab[(12 * i - startSlab) : (12 * i + stopSlab)] += zpiSlab

            # Assemble weak layer equations based on segment position and foundation
            if (pos == "l" or pos == "left") and k:
                # Left-most bedded segment
                zh0Weak[0:6, globalStartSlab : globalStartSlab + nConst] = zhiWeak[
                    0:6, :
                ]
                zp0Weak[0:6] += zpiWeak[0:6]

                if ki[i + 1]:
                    # Next segment is bedded
                    zh0Weak[6:18, globalStartSlab : globalStartSlab + nConst] = zhiWeak[
                        6:, :
                    ]
                    zp0Weak[6:18] += zpiWeak[6:]
                else:
                    # Next segment is free
                    zh0Weak[6:12, globalStartSlab : globalStartSlab + nConst] = zhiWeak[
                        12:, :
                    ]
                    zp0Weak[6:12] += zpiWeak[12:]

            elif (pos == "m" or pos == "mid") and k:
                # Middle bedded segment
                if kprev and ki[i + 1]:
                    # Both adjacent segments are bedded
                    zh0Weak[
                        globalStartWeak - 6 : globalStartWeak + 18,
                        globalStartSlab : globalStartSlab + nConst,
                    ] = zhiWeak
                    zp0Weak[globalStartWeak - 6 : globalStartWeak + 18] += zpiWeak
                elif kprev and not ki[i + 1]:
                    # Left bedded, right free
                    zh0Weak[
                        globalStartWeak - 6 : globalStartWeak + 12,
                        globalStartSlab : globalStartSlab + nConst,
                    ] = np.stack([zhiWeak[0:12, :], zhiWeak[18:, :]], axis=0)
                    zp0Weak[globalStartWeak - 6 : globalStartWeak + 12] += np.stack(
                        [zpiWeak[0:12], zpiWeak[18:]], axis=0
                    )
                elif not kprev and ki[i + 1]:
                    # Left free, right bedded
                    zh0Weak[
                        globalStartWeak : globalStartWeak + 18,
                        globalStartSlab : globalStartSlab + nConst,
                    ] = zhiWeak[6:, :]
                    zp0Weak[globalStartWeak : globalStartWeak + 18] += zpiWeak[6:]
                else:
                    # Both adjacent segments are free
                    zh0Weak[
                        globalStartWeak : globalStartWeak + 12,
                        globalStartSlab : globalStartSlab + nConst,
                    ] = np.stack([zhiWeak[6:12, :], zhiWeak[18:, :]], axis=0)
                    zp0Weak[globalStartWeak : globalStartWeak + 12] += np.stack(
                        [zpiWeak[6:12], zpiWeak[8:]], axis=0
                    )

            elif (pos == "r" or pos == "right") and k:
                # Right-most bedded segment
                zh0Weak[-6:, globalStartSlab : globalStartSlab + nConst] = zhiWeak[
                    -6:, :
                ]
                zp0Weak[-6:] += zpiWeak[-6:]

                if kprev:
                    # Previous segment is bedded
                    zh0Weak[-18:-6, globalStartSlab : globalStartSlab + nConst] = (
                        zhiWeak[0:12, :]
                    )
                    zp0Weak[-18:-6] += zpiWeak[0:12]
                else:
                    # Previous segment is free
                    zh0Weak[-12:-6, globalStartSlab : globalStartSlab + nConst] = (
                        zhiWeak[6:12, :]
                    )
                    zp0Weak[-12:-6] += zpiWeak[6:12]

            # Update global counters
            globalStartWeak += 12 if k else 0
            globalStartSlab += nConst
            kprev = k  # Store foundation state for next iteration

        # --- ASSEMBLE RIGHT-HAND SIDE --------------------------------------

        # Add loads at boundary segments
        for i, f in enumerate(fi[1:-1], start=1):
            rhsSlab[12 * i : 12 * i + 6] = np.array(f).reshape(-1, 1)

        if self.system not in ["skier"]:
            rhsSlab[0:6] = np.array(fi[0])  # Load at the left boundary of the strucutre
            rhsSlab[-6:] = np.array(
                fi[-1]
            )  # Load at the right boundary of the structure
        # Set boundary conditions for infinite systems
        if self.system not in ["skier-finite", "pst-", "-pst"]:
            # Left boundary
            rhsSlab[:6] = self.bc(
                self.zp(x=0, phi=phi, theta=theta, bed=ki[0], load=wi[0]),
                l=li[0],
                k=ki[0],
                pos="l",
            )[0:6]
            if ki[0]:
                rhsWeak[:6] = self.bc(
                    self.zp(x=0, phi=phi, theta=theta, bed=ki[0], load=wi[0]),
                    l=li[0],
                    k=ki[0],
                    pos="l",
                )[6:]

            # Right boundary
            rhsSlab[-6:] = self.bc(
                self.zp(x=li[-1], phi=phi, theta=theta, bed=ki[-1], load=wi[-1]),
                l=li[-1],
                k=ki[-1],
                pos="r",
            )[0:6]
            if ki[nS - 1]:
                rhsWeak[-6:] = self.bc(
                    self.zp(x=li[-1], phi=phi, theta=theta, bed=ki[-1], load=wi[-1]),
                    l=li[-1],
                    k=ki[-1],
                    pos="r",
                )[6:]
            if self.system in ["-vpst", "vpst-"]:
                # Calculate center of gravity and mass of
                # added or cut off slab segemen
                if theta != 0:
                    print(
                        "Error: " + self.system + " not implemented for biaxial bending"
                    )
                xs, zs, m = calc_vertical_bc_center_of_gravity(self.slab, phi)
                # Convert slope angle to radians
                phi = np.deg2rad(phi)
                # Translate inbto section forces and moments
                Nx = -self.g * m * np.sin(phi)
                My = -self.g * m * (xs * np.cos(phi) + zs * np.sin(phi))
                Vz = self.g * m * np.cos(phi)
                rhsSlab[:6] = np.vstack([Nx, 0, Vz, 0, My, 0])
                rhsSlab[-6:] = np.vstack([Nx, 0, Vz, 0, My, 0])

        # --- SOLVE SYSTEM -------------------------------------------------

        # Combine slab and weak layer equations
        zh0 = np.vstack([zh0Slab, zh0Weak])
        zp0 = np.vstack([zp0Slab, zp0Weak])
        rhs = np.vstack([rhsSlab, rhsWeak])

        # Solve for constants: zh0*C = rhs - zp0
        C = np.linalg.solve(zh0, rhs - zp0)

        # Reshape solution into matrix form
        CReturn = np.full((nS, nDOFbedded), np.nan, dtype=float)
        pos = 0
        for i in range(nS):
            if ki[i]:
                CReturn[i, :] = np.reshape(
                    C[pos : pos + nDOFbedded], C[pos : pos + nDOFbedded].shape[0]
                )
                pos += nDOFbedded
            else:
                CReturn[i, :nDOFfree] = np.reshape(
                    C[pos : pos + nDOFfree], C[pos : pos + nDOFfree].shape[0]
                )
                pos += nDOFfree

        return CReturn.T


class AnalysisMixin:
    """
    Mixin for the analysis of model outputs.

    Provides methods for the analysis of layered slabs on compliant
    elastic foundations.
    """

    def rasterize_solution(self, C, phi, theta, li, ki, wi, num=250, **kwargs):
        """
        Compute rasterized solution vector.

        Arguments
        ---------
        C : ndarray
            Vector of free constants.
        phi : float
            Inclination (degrees).
        li : ndarray
            List of segment lengths (mm).
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not.
        wi : ndarray
            List of booleans indicating wether segment is loaded with extra weights
        num : int
            Number of grid points.

        Returns
        -------
        xq : ndarray
            Grid point x-coordinates at which solution vector
            is discretized.
        zq : ndarray
            Matrix with solution vectors as colums at grid
            points xq.
        xb : ndarray
            Grid point x-coordinates that lie on a foundation.
        """
        # Unused arguments
        _ = kwargs

        # Drop zero-length segments
        isnonzero = li > 0
        C, ki, li = C[:, isnonzero], ki[isnonzero], li[isnonzero]

        # Compute number of plot points per segment (+1 for last segment)
        nq = np.ceil(li / li.sum() * num).astype("int")
        nq[ki] += 1

        # Provide cumulated length and plot point lists
        lic = np.insert(np.cumsum(li), 0, 0)
        nqc = np.insert(np.cumsum(nq), 0, 0)

        # Initialize arrays
        isbedded = np.full(nq.sum(), True)
        xq = np.full(nq.sum(), np.nan)
        zq = np.full([24, xq.size], np.nan)

        # Loop through segments
        for i, l in enumerate(li):
            # Get local x-coordinates of segment i
            xi = np.linspace(0, l, num=nq[i], endpoint=ki[i])
            # Compute start and end coordinates of segment i
            x0 = lic[i]
            # Assemble global coordinate vector
            xq[nqc[i] : nqc[i + 1]] = x0 + xi
            # Mask coordinates not on foundation (including endpoints)
            if not ki[i]:
                isbedded[nqc[i] : nqc[i + 1]] = False
            # Compute segment solution
            if ki[i]:
                zi = self.z(xi, C[:, [i]], l, phi, theta, ki[i], wi[i])
                zq[:, nqc[i] : nqc[i + 1]] = zi
            else:
                zi = self.z(xi, C[0:12, [i]], l, phi, theta, ki[i], wi[i])
                zq[0:12, nqc[i] : nqc[i + 1]] = zi
            # Assemble global solution matrix

        # Make sure cracktips are included
        transmissionbool = [ki[j] or ki[j + 1] for j, _ in enumerate(ki[:-1])]
        for i, truefalse in enumerate(transmissionbool, start=1):
            isbedded[nqc[i]] = truefalse

        # Assemble vector of coordinates on foundation
        xb = np.full(nq.sum(), np.nan)
        xb[isbedded] = xq[isbedded]

        return xq, zq, xb

    def ginc(self, C0, C1, phi, theta, li, ki, wi, k0, **kwargs):
        """
        Compute incremental energy relase rate of of all cracks.

        Arguments
        ---------
        C0 : ndarray
            Free constants of uncracked solution.
        C1 : ndarray
            Free constants of cracked solution.
        phi : float
            Inclination (degress).
        li : ndarray
            List of segment lengths.
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the cracked configuration.
        k0 : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the uncracked configuration.

        Returns
        -------
        ndarray
            List of total, mode I, and mode II energy release rates.
        """
        # Unused arguments
        _ = kwargs

        # Make sure inputs are np.arrays
        li, ki, k0, wi = np.array(li), np.array(ki), np.array(k0), np.array(wi)

        # Reduce inputs to segments with crack advance
        iscrack = k0 & ~ki
        C0, C1, li = C0[:, iscrack], C1[:, iscrack], li[iscrack]

        # Compute total crack lenght and initialize outputs
        da = li.sum() if li.sum() > 0 else np.nan
        Ginc1, Ginc2 = 0, 0

        # Loop through segments with crack advance
        for j, l in enumerate(li):
            # Uncracked (0) and cracked (1) solutions at integration points
            z0 = partial(
                self.z, C=C0[:, [j]], l=l, phi=phi, theta=theta, bed=True, load=wi[j]
            )
            z1 = partial(
                self.z, C=C1[:, [j]], l=l, phi=phi, theta=theta, bed=False, load=wi[j]
            )

            # Mode I (1) and II (2) integrands at integration points
            int1 = partial(self.int1, z0=z0, z1=z1)
            int2 = partial(self.int2, z0=z0, z1=z1)

            # Segement contributions to total crack opening integral
            Ginc1 += quad(int1, 0, l, epsabs=self.tol, epsrel=self.tol) / (2 * da)
            Ginc2 += quad(int2, 0, l, epsabs=self.tol, epsrel=self.tol) / (2 * da)

        return np.array([Ginc1 + Ginc2, Ginc1, Ginc2]).flatten()

    def gdif(self, C, phi, theta, li, ki, wi, unit="kJ/m^2", **kwargs):
        """
        Compute differential energy release rate of all crack tips.

        Arguments
        ---------
        C : ndarray
            Free constants of the solution.
        phi : float
            Inclination (degress).
        li : ndarray
            List of segment lengths.
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the cracked configuration.

        Returns
        -------
        ndarray
            List of total, mode I, and mode II energy release rates.
        """
        # Unused arguments
        _ = kwargs

        # Get number and indices of segment transitions
        ntr = len(li) - 1
        itr = np.arange(ntr)

        # Identify bedded-free and free-bedded transitions as crack tips
        # Differ between bedded->free and free->bedded !!!!
        iscracktip = [ki[j] != ki[j + 1] for j in range(ntr)]

        # Transition indices of crack tips and total number of crack tips
        ict = itr[iscracktip]
        nct = len(ict)
        # Initialize energy release rate array
        Gdif = np.zeros([4, nct])
        #
        # Compute energy relase rate of all crack tips
        for j, idx in enumerate(ict):
            # Solution at crack tip

            if ki[idx]:
                ztip = self.z(
                    li[idx], C[:, [idx]], li[idx], phi, theta, bed=ki[idx], load=wi[idx]
                )
                zback = self.z(
                    0 * li[idx],
                    C[:, [idx]],
                    li[idx],
                    phi,
                    theta,
                    bed=ki[idx],
                    load=wi[idx],
                )
            else:
                ztip = self.z(
                    0,
                    C[:, [idx + 1]],
                    li[idx + 1],
                    phi,
                    theta,
                    bed=ki[idx + 1],
                    load=wi[idx + 1],
                )
                zback = self.z(
                    1 * li[idx + 1],
                    C[:, [idx + 1]],
                    li[idx + 1],
                    phi,
                    theta,
                    bed=ki[idx + 1],
                    load=wi[idx + 1],
                )
            # Mode I and II differential energy release rates
            Gdif[1:, j] = (
                self.Gi(ztip, zback, unit=unit)[0],
                self.Gii(ztip, zback, unit=unit)[0],
                self.Giii(ztip, zback, unit=unit)[0],
            )

        # Sum mode I and II contributions
        Gdif[0, :] = Gdif[1, :] + Gdif[2, :] + Gdif[3, :]

        # Adjust contributions for center cracks
        if nct > 1:
            avgmask = np.full(nct, True)  # Initialize mask
            avgmask[[0, -1]] = ki[[0, -1]]  # Do not weight edge cracks
            Gdif[:, avgmask] *= 0.5  # Weigth with half crack length

        # Return total differential energy release rate of all crack tips
        return Gdif.sum(axis=1)

    def get_zmesh(self, dz=2):
        """
        Get z-coordinates of grid points and corresponding elastic properties.

        Arguments
        ---------
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.

        Returns
        -------
        mesh : ndarray
            Mesh along z-axis. Columns are a list of z-coordinates (mm) of
            grid points along z-axis with at least two grid points (top,
            bottom) per layer, Young's modulus of each grid point, shear
            modulus of each grid point, and Poisson's ratio of each grid
            point.
        """
        # Get ply (layer) coordinates
        z = self.get_ply_coordinates()
        # Compute number of grid points per layer
        nlayer = np.ceil((z[1:] - z[:-1]) / dz).astype(np.int32) + 1
        # Calculate grid points as list of z-coordinates (mm)
        zi = np.hstack(
            [np.linspace(z[i], z[i + 1], n, endpoint=True) for i, n in enumerate(nlayer)]
        )
        # Get lists of corresponding elastic properties (E, nu, rho)
        si = np.repeat(self.slab[:, [2, 4, 0]], nlayer, axis=0)
        # Assemble mesh with columns (z, E, G, nu)
        return np.column_stack([zi, si])

    def Sxx(self, Z, phi, dz=2, unit="kPa"):
        """
        Compute axial normal stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray, float
            Axial slab normal stress in specified unit.
        """
        # Unit conversion dict
        convert = {"kPa": 1e3, "MPa": 1}

        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh[:, 0]
        rho = 1e-12 * zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Initialize axial normal stress Sxx
        Sxx = np.zeros(shape=[n, m])

        # Compute axial normal stress Sxx at grid points in MPa
        for i, (z, E, nu, _) in enumerate(zmesh):
            Sxx[i, :] = E / (1 - nu**2) * self.du_dx(Z, z)

        # Calculate weight load at grid points and superimpose on stress field
        qt = -rho * self.g * np.sin(np.deg2rad(phi))
        for i, qi in enumerate(qt[:-1]):
            Sxx[i, :] += qi * (zi[i + 1] - zi[i])
        Sxx[-1, :] += qt[-1] * (zi[-1] - zi[-2])

        # Return axial normal stress in specified unit
        return convert[unit] * Sxx

    def Txz(self, Z, phi, dz=2, unit="kPa"):
        """
        Compute shear stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray
            Shear stress at grid points in the slab in specified unit.
        """
        # Unit conversion dict
        convert = {"kPa": 1e3, "MPa": 1}
        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh[:, 0]
        rho = 1e-12 * zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Get second derivatives of centerline displacement u0 and
        # cross-section rotaiton psi of all grid points along the x-axis
        du0_dxdx = self.du0_dxdx(Z, phi)
        dpsi_dxdx = self.dpsi_dxdx(Z, phi)

        # Initialize first derivative of axial normal stress sxx w.r.t. x
        dsxx_dx = np.zeros(shape=[n, m])

        # Calculate first derivative of sxx at z-grid points
        for i, (z, E, nu, _) in enumerate(zmesh):
            dsxx_dx[i, :] = E / (1 - nu**2) * (du0_dxdx + z * dpsi_dxdx)

        # Calculate weight load at grid points
        qt = -rho * self.g * np.sin(np.deg2rad(phi))

        # Integrate -dsxx_dx along z and add cumulative weight load
        # to obtain shear stress Txz in MPa
        Txz = cumulative_trapezoid(dsxx_dx, zi, axis=0, initial=0)
        Txz += cumulative_trapezoid(qt, zi, initial=0)[:, None]

        # Return shear stress Txz in specified unit
        return convert[unit] * Txz

    def Szz(self, Z, phi, dz=2, unit="kPa"):
        """
        Compute transverse normal stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray, float
            Transverse normal stress at grid points in the slab in
            specified unit.
        """
        # Unit conversion dict
        convert = {"kPa": 1e3, "MPa": 1}

        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh[:, 0]
        rho = 1e-12 * zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Get third derivatives of centerline displacement u0 and
        # cross-section rotaiton psi of all grid points along the x-axis
        du0_dxdxdx = self.du0_dxdxdx(Z, phi)
        dpsi_dxdxdx = self.dpsi_dxdxdx(Z, phi)

        # Initialize second derivative of axial normal stress sxx w.r.t. x
        dsxx_dxdx = np.zeros(shape=[n, m])

        # Calculate second derivative of sxx at z-grid points
        for i, (z, E, nu, _) in enumerate(zmesh):
            dsxx_dxdx[i, :] = E / (1 - nu**2) * (du0_dxdxdx + z * dpsi_dxdxdx)

        # Calculate weight load at grid points
        qn = rho * self.g * np.cos(np.deg2rad(phi))

        # Integrate dsxx_dxdx twice along z to obtain transverse
        # normal stress Szz in MPa
        integrand = cumulative_trapezoid(dsxx_dxdx, zi, axis=0, initial=0)
        Szz = cumulative_trapezoid(integrand, zi, axis=0, initial=0)
        Szz += cumulative_trapezoid(-qn, zi, initial=0)[:, None]

        # Return shear stress txz in specified unit
        return convert[unit] * Szz

    def principal_stress_slab(
        self, Z, phi, dz=2, unit="kPa", val="max", normalize=False
    ):
        """
        Compute maxium or minimum principal stress in slab layers.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.
        val : str, optional
            Maximum 'max' or minimum 'min' principal stress. Default is 'max'.
        normalize : bool
            Toggle layerwise normalization to strength.

        Returns
        -------
        ndarray
            Maximum or minimum principal stress in specified unit.

        Raises
        ------
        ValueError
            If specified principal stress component is neither 'max' nor
            'min', or if normalization of compressive principal stress
            is requested.
        """
        # Raise error if specified component is not available
        if val not in ["min", "max"]:
            raise ValueError(f"Component {val} not defined.")

        # Multiplier selection dict
        m = {"max": 1, "min": -1}

        # Get axial normal stresses, shear stresses, transverse normal stresses
        Sxx = self.Sxx(Z=Z, phi=phi, dz=dz, unit=unit)
        Txz = self.Txz(Z=Z, phi=phi, dz=dz, unit=unit)
        Szz = self.Szz(Z=Z, phi=phi, dz=dz, unit=unit)

        # Calculate principal stress
        Ps = (Sxx + Szz) / 2 + m[val] * np.sqrt((Sxx - Szz) ** 2 + 4 * Txz**2) / 2

        # Raise error if normalization of compressive stresses is attempted
        if normalize and val == "min":
            raise ValueError("Can only normlize tensile stresses.")

        # Normalize tensile stresses to tensile strength
        if normalize and val == "max":
            # Get layer densities
            rho = self.get_zmesh(dz=dz)[:, 3]
            # Normlize maximum principal stress to layers' tensile strength
            return Ps / tensile_strength_slab(rho, unit=unit)[:, None]

        # Return absolute principal stresses
        return Ps

    def principal_stress_weaklayer(
        self, Z, sc=2.6, unit="kPa", val="min", normalize=False
    ):
        """
        Compute maxium or minimum principal stress in the weak layer.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        sc : float
            Weak-layer compressive strength. Default is 2.6 kPa.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.
        val : str, optional
            Maximum 'max' or minimum 'min' principal stress. Default is 'min'.
        normalize : bool
            Toggle layerwise normalization to strength.

        Returns
        -------
        ndarray
            Maximum or minimum principal stress in specified unit.

        Raises
        ------
        ValueError
            If specified principal stress component is neither 'max' nor
            'min', or if normalization of tensile principal stress
            is requested.
        """
        # Raise error if specified component is not available
        if val not in ["min", "max"]:
            raise ValueError(f"Component {val} not defined.")

        # Multiplier selection dict
        m = {"max": 1, "min": -1}

        # Get weak-layer normal and shear stresses
        sig = self.sig(Z, unit=unit)
        tau = self.tau(Z, unit=unit)

        # Calculate principal stress
        ps = sig / 2 + m[val] * np.sqrt(sig**2 + 4 * tau**2) / 2

        # Raise error if normalization of tensile stresses is attempted
        if normalize and val == "max":
            raise ValueError("Can only normlize compressive stresses.")

        # Normalize compressive stresses to compressive strength
        if normalize and val == "min":
            return ps / sc

        # Return absolute principal stresses
        return ps


class OutputMixin:
    """
    Mixin for outputs.

    Provides convenience methods for the assembly of output lists
    such as rasterized displacements or rasterized stresses.
    """

    def external_potential(self, C, phi, L, **segments):
        """
        Compute total external potential (pst only).

        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi_ext : float
            Total external potential (Nmm).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(C=C, phi=phi, **segments)
        _ = xq, xb
        # Compute displacements where weight loads are applied
        w0 = self.w(zq)
        us = self.u(zq, z0=self.zs)
        # Get weight loads
        qn = self.calc_qn()
        qt = self.calc_qt()
        # use +/- and us[0]/us[-1] according to system and phi
        # compute total external potential
        Pi_ext = (
            -qn * (segments["li"][0] + segments["li"][1]) * np.average(w0)
            - qn * (L - (segments["li"][0] + segments["li"][1])) * self.tc
        )
        # Ensure
        if self.system in ["pst-"]:
            ub = us[-1]
        elif self.system in ["-pst"]:
            ub = us[0]
        Pi_ext += (
            -qt * (segments["li"][0] + segments["li"][1]) * np.average(us)
            - qt * (L - (segments["li"][0] + segments["li"][1])) * ub
        )
        if self.system not in ["pst-", "-pst"]:
            print("Input error: Only pst-setup implemented at the moment.")

        return Pi_ext

    def internal_potential(self, C, phi, L, **segments):
        """
        Compute total internal potential (pst only).

        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi_int : float
            Total internal potential (Nmm).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(C=C, phi=phi, **segments)

        # Compute section forces
        N, M, V = self.N(zq), self.M(zq), self.V(zq)

        # Drop parts of the solution that are not a foundation
        zweak = zq[:, ~np.isnan(xb)]
        xweak = xb[~np.isnan(xb)]

        # Compute weak layer displacements
        wweak = self.w(zweak)
        uweak = self.u(zweak, z0=self.h / 2)

        # Compute stored energy of the slab (monte-carlo integration)
        n = len(xq)
        nweak = len(xweak)
        # energy share from moment, shear force, wl normal and tangential springs
        Pi_int = (
            L / 2 / n / self.A11 * np.sum([Ni**2 for Ni in N])
            + L
            / 2
            / n
            / (self.D11 - self.B11**2 / self.A11)
            * np.sum([Mi**2 for Mi in M])
            + L / 2 / n / self.kA55 * np.sum([Vi**2 for Vi in V])
            + L * self.kn / 2 / nweak * np.sum([wi**2 for wi in wweak])
            + L * self.kt / 2 / nweak * np.sum([ui**2 for ui in uweak])
        )
        # energy share from substitute rotation spring
        if self.system in ["pst-"]:
            Pi_int += 1 / 2 * M[-1] * (self.psi(zq)[-1]) ** 2
        elif self.system in ["-pst"]:
            Pi_int += 1 / 2 * M[0] * (self.psi(zq)[0]) ** 2
        else:
            print("Input error: Only pst-setup implemented at the moment.")

        return Pi_int

    def total_potential(self, C, phi, L, **segments):
        """
        Returns total differential potential

        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi : float
            Total differential potential (Nmm).
        """
        Pi_int = self.internal_potential(C, phi, L, **segments)
        Pi_ext = self.external_potential(C, phi, L, **segments)

        return Pi_int + Pi_ext

    def get_weaklayer_shearstress(self, x, z, unit="MPa", removeNaNs=False):
        """
        Compute weak-layer shear stress.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of unsupported
            (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        unit : {'MPa', 'kPa'}, optional
            Stress output unit. Default is MPa.
        keepNaNs : bool
            If set, do not remove

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        sig : ndarray
            Normal stress (stress unit input).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        x = x / 10
        tau = self.tau(z, unit=unit)
        # Filter stresses in unspupported segments
        if removeNaNs:
            # Remove coordinate-stress pairs where no weak layer is present
            tau = tau[~np.isnan(x)]
            x = x[~np.isnan(x)]
        else:
            # Set stress NaN where no weak layer is present
            tau[np.isnan(x)] = np.nan

        return x, tau

    def get_weaklayer_normalstress(self, x, z, unit="MPa", removeNaNs=False):
        """
        Compute weak-layer normal stress.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of unsupported
            (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        unit : {'MPa', 'kPa'}, optional
            Stress output unit. Default is MPa.
        keepNaNs : bool
            If set, do not remove

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        sig : ndarray
            Normal stress (stress unit input).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        x = x / 10
        sig = self.sig(z, unit=unit)
        # Filter stresses in unspupported segments
        if removeNaNs:
            # Remove coordinate-stress pairs where no weak layer is present
            sig = sig[~np.isnan(x)]
            x = x[~np.isnan(x)]
        else:
            # Set stress NaN where no weak layer is present
            sig[np.isnan(x)] = np.nan

        return x, sig

    def get_slab_displacement(self, x, z, loc="mid", unit="mm"):
        """
        Compute horizontal slab displacement.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        loc : {'top', 'mid', 'bot'}
            Get displacements of top, midplane or bottom of slab.
            Default is mid.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Displacement output unit. Default is mm.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Horizontal displacements (unit input).
        """
        # Coordinates (cm)
        x = x / 10
        # Locator
        z0 = {"top": -self.h / 2, "mid": 0, "bot": self.h / 2}
        # Displacement (unit)
        u = self.u(z, z0=z0[loc], unit=unit)
        # Output array
        return x, u

    def get_slab_deflection(self, x, z, unit="mm"):
        """
        Compute vertical slab displacement.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
            Default is mid.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Displacement output unit. Default is mm.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Vertical deflections (unit input).
        """
        # Coordinates (cm)
        x = x / 10
        # Deflection (unit)
        w = self.w(z, unit=unit)
        # Output array
        return x, w

    def get_slab_rotation(self, x, z, unit="degrees"):
        """
        Compute slab cross-section rotation angle.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
            Default is mid.
        unit : {'deg', degrees', 'rad', 'radians'}, optional
            Rotation angle output unit. Default is degrees.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Cross section rotations (unit input).
        """
        # Coordinates (cm)
        x = x / 10
        # Cross-section rotation angle (unit)
        psi = self.psi(z, unit=unit)
        # Output array
        return x, psi
