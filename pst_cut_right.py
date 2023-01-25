"""Driver Code for Weac - PST system with touchdown and calculation of the energy release rate"""
# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Project imports
import weac

# Functions
def central_diff(values):
    """Numerically differentiate a series of discrete values using the central difference method."""
    n = len(values)
    derivatives = []
    for j in range(1, n-1):
        # Use the central difference method to approximate the derivative at each point
        derivative = (values[j+1] - values[j-1])/da/2
        derivatives.append(derivative)
    return derivatives

def plot_data(x, y1, y2):
    """Plots two series of y-values over one series of x-values"""
    plt.plot(x, y1, label='Gweac')
    plt.plot(x, y2, label='Gdif')
    plt.xlabel('cracklength')
    plt.ylabel('energy release rate')
    plt.legend()
    plt.show()
    #plt.tight_layout()
    #plt.savefig('scratch/energy_release_rates.png')
    #plt.close()

# System characteristics
rho = 200                                   # Slab density (kg/m3)
height = 180                                # Slab height (mm)
totallength = 5000                          # Total length (mm)
da = 50.0                                   # Crack length increment
cracklength = np.arange(da, 80*da, da)       # Crack length (mm)
inclination = 0                             # Slope inclination (°); pos angle downslope cut in pst-
myprofile = [[rho,height]]                  # Slab layering

# Initiante Gdif and Pi
Gdif = np.array([])
Pi = np.array([])

# Solve system
for i in cracklength:
    # Create model instance
    pst_cut_right = weac.Layered(system='pst-', layers=myprofile, \
            a=float(i), cf=1.0, ratio=16, phi=inclination)
    # Obtain lists of segment lengths, locations of foundations, and position
    seg_pst = pst_cut_right.calc_segments(
        L=totallength)['crack']
    # Assemble system of linear equations and solve the boundary-value problem for free constants.
    C_pst = pst_cut_right.assemble_and_solve(
        phi=inclination, **seg_pst)
    # Prepare the output by rasterizing the solution vector at all horizontal positions xsl (slab)
    xsl_pst, z_pst, xwl_pst = pst_cut_right.rasterize_solution(
        C=C_pst, phi=inclination, num=totallength/10, **seg_pst)
    # Calculate energy release rate
    Gdif = np.append(Gdif, pst_cut_right.gdif(C_pst, inclination,**seg_pst)[0])
    Pi = np.append(Pi, pst_cut_right.total_potential(C_pst, phi=inclination, L=totallength, **seg_pst))

# Calc energy release rates
G = central_diff(-Pi)
# Plot energy release rates
plot_data(cracklength[1:-1],G,Gdif[1:-1])
