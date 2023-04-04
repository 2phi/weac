"""Driver Code for Weac - PST system with touchdown and calculation of the energy release rate"""
# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Project imports
import weac

# Functions
def plot_data(x, y1, name):
    plt.plot(x, y1, label=name[:-4])
    plt.xlabel(r'$\mathrm{crack\ length\ a\ (mm)}$')
    plt.ylabel(r'$\mathrm{touchdown\ length\ \lambda\ (mm)}$')
    plt.legend()
    #plt.show()
    plt.tight_layout()
    plt.savefig('scratch/'+name)
    plt.close()

# System characteristics
PST = '190208_BUN_PST1'
rho = 247                                       # Slab density (kg/m3)
height = 940                                    # Slab height (mm)
totallength = 8500                              # Total length (mm)
da = 100.0                                      # Crack length increment
cracklength = np.arange(0.0,totallength,da)     # Crack length (mm)
ccl = 770
cal = 3400
inclination = 0                                 # Slope inclination (°); pos angle downslope cut in pst-
myprofile = [[rho,height]]                      # Slab layering

# Initiante postprocessing arrays
Gdif = np.array([])
#Pi = np.array([])
td = np.array([])

for i in cracklength:
    # Create model instance
    pst_cut_right = weac.Layered(system='-pst', layers=myprofile, \
            L=totallength, a=float(i), cf=1.0, phi=inclination)
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
    #Pi = np.append(Pi, pst_cut_right.total_potential(C_pst, phi=inclination, L=totallength, **seg_pst))
    td = np.append(td, pst_cut_right.td)
    # Plot contour
    #weac.plot.contours(pst_cut_right, x=xsl_pst, z=z_pst, i=i, window=totallength, scale=50)
    # Plot displacements
    #weac.plot.displacements(pst_cut_right, x=xsl_pst, z=z_pst, i=i, **seg_pst)
    # Plot stresses
    #weac.plot.stresses(pst_cut_right, x=xsl_pst, z=z_pst, i=i, **seg_pst)

# Plot energy release rates
plot_data(cracklength[1:-1],Gdif[1:-1],PST+'_G.png')
# Plot touchdown length
plot_data(cracklength,td,PST+'_td.png')

# Save data to file
data_td = np.column_stack((cracklength,td))
np.savetxt('scratch/'+PST+'_td.txt', data_td)
data_G = np.column_stack((cracklength,Gdif))
np.savetxt('scratch/'+PST+'_G.txt', data_G)
