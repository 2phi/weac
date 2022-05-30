# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import c_style_comment
import sys
sys.path.append("..")

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
# myprofile = 'a'

# === CREATE MODEL INSTANCES ==========================================

# One skier on homogeneous default slab (240 kg/m^3, 200 mm)
skier = weac.Layered(system='skier', layers='medium')

# === INSPECT LAYERING ================================================

weac.plot.slab_profile(skier)

# === SKIER-INDUCED ===================================================

# Example with two segements, one skier load
# (between segments 1 & 2) and no crack.

#                   |
#                   v
# +-----------------+-----------------+
# |                 |                 |
# |        1        |        2        |
# |                 |                 |
# +-----------------+-----------------+
#  |||||||||||||||||||||||||||||||||||
# --------------------------------------

# Input
totallength = 1e4                       # Total length (mm)
cracklength = 0                         # Crack length (mm)
inclination = 0                        # Slope inclination (°)
skierweight = 100                        # Skier weigth (kg)

# Obtain lists of segment lengths, locations of foundations,
# and position and magnitude of skier loads from inputs. We
# can choose to analyze the situtation before a crack appears
# even if a cracklength > 0 is set by replacing the 'crack'
# key thorugh the 'nocrack' key.
seg_skier = skier.calc_segments(
    L=totallength, a=cracklength, m=skierweight)['nocrack']

# Assemble system of linear equations and solve the
# boundary-value problem for free constants.
C_skier = skier.assemble_and_solve(
    phi=inclination, **seg_skier)

# Prepare the output by rasterizing the solution vector at all
# horizontal positions xsl (slab). The result is returned in the
# form of the ndarray z. Also provides xwl (weak layer) that only
# contains x-coordinates that are supported by a foundation.
xq_skier, zq_skier, xb_skier = skier.rasterize_solution(
    C=C_skier, phi=inclination, **seg_skier)

# === VISUALIZE SLAB DEFORMATIONS / DISPLACEMENTS / WEAK LAYER STRESSES =====================================
plot = 1
if plot:
    weac.plot.contours(skier, x=xq_skier, z=zq_skier, window=200, scale=100)
    weac.plot.displacements(skier, x=xq_skier, z=zq_skier, **seg_skier)
    weac.plot.stresses(skier, x=xb_skier, z=zq_skier, **seg_skier)
