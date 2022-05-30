# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import c_style_comment
import sys
#sys.path.append("..")

# Project imports
import weac

# === DEFINE SLAB LAYERING ============================================

# Either use custom profile
myprofile = [[170, 100],  # (1) surface layer
             [190,  40],  # (2) 2nd layer
             [230, 130],  #  :
             [250,  20],  #  :
             [210,  70],  # (i) i-th layer
             [380,  20],  #  :
             [280, 100]]  # (N) last slab layer above weak layer

# Or select a predefined profile from database
myprofile = 'medium'

# === CREATE MODEL INSTANCES ==========================================

# Propagation saw test cut from the right side with custom layering
pst_cut_right = weac.Layered(system='pst-', layers=myprofile)


# === INSPECT LAYERING ================================================

weac.plot.slab_profile(pst_cut_right)

# Example with a crack cut from the right-hand side.

# +-----------------------------+-----+
# |                             |     |
# |             1               |  2  |
# |                             |     |
# +-----------------------------+-----+
#  |||||||||||||||||||||||||||||
# --------------------------------------

# Input
totallength = 2000                      # Total length (mm)
cracklength = 300                       # Crack length (mm)
inclination = 20                       # Slope inclination (°)

# Obtain lists of segment lengths, locations of foundations,
# and position and magnitude of skier loads from inputs. We
# can choose to analyze the situtation before a crack appears
# even if a cracklength > 0 is set by replacing the 'crack'
# key thorugh the 'nocrack' key.
seg_pst = pst_cut_right.calc_segments(
    L=totallength, a=cracklength)['nocrack']

# Assemble system of linear equations and solve the
# boundary-value problem for free constants.

C_pst = pst_cut_right.assemble_and_solve(
    phi=inclination, **seg_pst)

# Prepare the output by rasterizing the solution vector at all
# horizontal positions xsl (slab). The result is returned in the
# form of the ndarray z. Also provides xwl (weak layer) that only
# contains x-coordinates that are supported by a foundation.
xsl_pst, z_pst, xwl_pst = pst_cut_right.rasterize_solution(
    C=C_pst, phi=inclination, **seg_pst)

# === VISUALIZE SLAB DEFORMATIONS / DISPLACEMENTS / WEAK LAYER STRESSES =====================================
plot = 1
if plot:
    weac.plot.contours(pst_cut_right, x=xsl_pst, z=z_pst, scale=100)
    weac.plot.displacements(pst_cut_right, x=xsl_pst, z=z_pst, **seg_pst)
    weac.plot.stresses(pst_cut_right, x=xwl_pst, z=z_pst, **seg_pst)
