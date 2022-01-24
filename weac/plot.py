"""Plotting resources for the WEak Layer AntiCrack nucleation model."""
# pylint: disable=invalid-name,too-many-locals,too-many-branches
# pylint: disable=too-many-arguments,too-many-statements

# Third party imports
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt

# Project imports
from weac.tools import isnotebook

# === SET PLOT STYLES =========================================================


def set_plotstyles():
    """Define styles plot markers, labels and colors."""
    labelstyle = {         # Text style of plot labels
        'backgroundcolor': 'w',
        'horizontalalignment': 'center',
        'verticalalignment': 'center'}
    # markerstyle = {        # Style of plot markers
    #     'marker': 'o',
    #     'markersize': 5,
    #     'markerfacecolor': 'w',
    #     'zorder': 3}
    colors = np.array([    # TUD color palette
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
    return labelstyle, colors


# === CONVENIENCE FUNCTIONS ===================================================


class MidpointNormalize(Normalize):
    """Colormap normalization to a specified midpoint. Default is 0."""

    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        """Inizialize normalization."""
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """Make instances callable as functions."""
        normalized_min = max(0, 0.5*(1 - abs(
            (self.midpoint - self.vmin)/(self.midpoint - self.vmax))))
        normalized_max = min(1, 0.5*(1 + abs(
            (self.vmax - self.midpoint)/(self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def outline(grid):
    """Extract outline values of a 2D array (matrix, grid)."""
    top = grid[0, :-1]
    right = grid[:-1, -1]
    bot = grid[-1, :0:-1]
    left = grid[::-1, 0]

    return np.hstack([top, right, bot, left])


# === PLOT SLAB PROFILE =======================================================


def slab_profile(instance):
    """Create bar chart of slab profile."""
    # Plot Setup
    plt.rcdefaults()
    plt.rc('font', family='serif', size=10)
    plt.rc('mathtext', fontset='cm')

    # Create figure
    fig = plt.figure(figsize=(8/3, 4))
    ax1 = fig.gca()

    # Initialize coordinates
    x = []
    y = []
    total_heigth = 0

    for line in np.flipud(instance.slab):
        x.append(line[0])
        x.append(line[0])

        y.append(total_heigth)
        total_heigth = total_heigth + line[1]
        y.append(total_heigth)

    # Set axis labels
    ax1.set_xlabel(r'$\longleftarrow$ Density $\rho$ (kg/m$^3$)')
    ax1.set_ylabel(r'Height above weak layer (mm) $\longrightarrow$')

    ax1.set_xlim(500, 0)

    ax1.fill_betweenx(y, 0, x)
    plt.show()

    # Reset plot styles
    plt.rcdefaults()

# === DEFORMATION CONTOUR PLOT ================================================


def contours(instance, x, z, window=1e12, scale=100):
    """
    Plot 2D deformation contours.

    Arguments
    ---------
    instance : object
        Instance of layered class.
    x : ndarray
        Discretized x-coordinates (mm).
    z : ndarray
        Solution vectors at positions x as columns of matrix z.
    window : int
        Plot window (cm) around maximum vertical deflection.
    scale : int
        Scaling factor for the visualization of displacements.
    """
    # Plot Setup
    plt.rcdefaults()
    plt.rc('font', family='serif', size=10)
    plt.rc('mathtext', fontset='cm')

    # Calculate top-to-bottom vertical positions (mm) in beam coordinate system
    y = np.linspace(-instance.h/2, instance.h/2, num=21)

    # Compute displacements on grid (cm)
    U = 1e-1*np.vstack([instance.u(z, z0=y, unit='mm') for y in y])
    W = 1e-1*np.vstack([instance.w(z, unit='mm') for _ in y])

    # Compute grid coordinates with vertical origin at top surface (cm)
    X, Y = np.meshgrid(1e-1*x, 1e-1*(y + instance.h/2))

    # Plot outline
    plt.plot(outline(X), outline(Y), 'k--', alpha=0.3, linewidth=1)
    plt.plot(outline(X+scale*U), outline(Y+scale*W), 'k', linewidth=1)

    # Get x-coordinate of maximum deflection w (cm) and derive plot limits
    xfocus = x[np.max(np.argmax(W, axis=1))]/10
    xmax = np.min([np.max(X+scale*U), xfocus + window/2])
    xmin = np.max([np.min(X+scale*U), xfocus - window/2])

    # Normalize colormap
    norm = MidpointNormalize(vmin=1e3*np.min(U), vmax=1e3*np.max(U))

    # Plot contours on deformed grid
    plt.contourf(X+scale*U, Y+scale*W, 1e3*U,
                 norm=norm, cmap='RdBu_r', levels=100, alpha=0.2)

    # Plot setup
    plt.axis('scaled')
    plt.xlim([xmin, xmax])
    plt.gca().set_aspect(2)
    plt.gca().invert_yaxis()
    plt.gca().use_sticky_edges = False

    # Plot labels
    plt.gca().set_xlabel(r'lateral position $x$ (cm) $\longrightarrow$')
    plt.gca().set_ylabel('depth below surface\n' r'$\longleftarrow $ $d$ (cm)')
    plt.title(fr'${scale}\!\times\!$ scaled deformations (cm)', size=10)

    # Colorbar
    cbar = plt.colorbar(
        shrink=0.5,
        label=('lateral displacements\n'
               r'$u$ ($\mu\mathrm{m}$) $\longrightarrow$'))
    cbar.ax.tick_params(labelsize=8)

    # Reset plot styles
    plt.rcdefaults()


# === BASE PLOT FUNCTION ======================================================


def plot_data(
        name, ax1data, ax1label,
        ax2data=None, ax2label=None,
        labelpos=None, vlines=True,
        li=False, mi=False, ki=False,
        xlabel=r'Horizontal position $x$ (cm)'):
    """Plot data. Base function."""
    # Figure setup
    plt.rcdefaults()
    plt.rc('font', family='serif', size=10)
    plt.rc('mathtext', fontset='cm')

    # Plot styles
    labelstyle, colors = set_plotstyles()

    # Create figure
    fig = plt.figure(figsize=(4, 8/3))
    ax1 = fig.gca()

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
        labelpos = int(0.95*len(x[~np.isnan(x)]))

    # Fill left y-axis
    i = 0
    for x, y, label in ax1data:
        i += 1
        if label == '' or 'FEA' in label:
            # line, = ax1.plot(x, y, 'k:', linewidth=1)
            ax1.plot(x, y, linewidth=3, color='white')
            line, = ax1.plot(x, y, ':', linewidth=1)  # , color='black'
            thislabelpos = -2
            x, y = x[~np.isnan(x)], y[~np.isnan(x)]
            xtx = (x[thislabelpos - 1] + x[thislabelpos])/2
            ytx = (y[thislabelpos - 1] + y[thislabelpos])/2
            ax1.text(xtx, ytx, label, color=line.get_color(),
                     **labelstyle)
        else:
            # Plot line
            ax1.plot(x, y, linewidth=3, color='white')
            line, = ax1.plot(x, y, linewidth=1)
            # Line label
            x, y = x[~np.isnan(x)], y[~np.isnan(x)]
            if len(x) > 0:
                xtx = (x[labelpos - 10*i - 1] + x[labelpos - 10*i])/2
                ytx = (y[labelpos - 10*i - 1] + y[labelpos - 10*i])/2
                ax1.text(xtx, ytx, label, color=line.get_color(),
                         **labelstyle)

    # Fill right y-axis
    if ax2data:
        # Create right y-axis
        ax2 = ax1.twinx()
        # Set axis label
        ax2.set_ylabel(ax2label + r' $\longrightarrow$')
        # Fill
        for x, y, label in ax2data:
            # Plot line
            ax2.plot(x, y, linewidth=3, color='white')
            line, = ax2.plot(x, y, linewidth=1, color=colors[8, 0])
            # Line label
            x, y = x[~np.isnan(x)], y[~np.isnan(x)]
            xtx = (x[labelpos - 1] + x[labelpos])/2
            ytx = (y[labelpos - 1] + y[labelpos])/2
            ax2.text(xtx, ytx, label, color=line.get_color(),
                     **labelstyle)

    # Save figure
    filename = name + '.pdf'
    if isnotebook():
        plt.show()
    else:
        print('Rendering', filename, '...')
        plt.savefig('plots/' + filename, bbox_inches='tight')

    # Reset plot styles
    plt.rcdefaults()


# === PLOT WRAPPERS ===========================================================


def displacements(instance, x, z, **segments):
    """Wrap for dispalcements plot."""
    data = [
        [x/10, instance.u(z, z0=0, unit='mm'), r'$u_0\ (\mathrm{mm})$'],
        [x/10, -instance.w(z, unit='mm'), r'$-w\ (\mathrm{mm})$'],
        [x/10, instance.psi(z, unit='degrees'), r'$\psi\ (^\circ)$ ']
    ]
    plot_data(ax1label=r'Displacements', ax1data=data,
              name='disp', **segments)


def section_forces(instance, x, z, **segments):
    """Wrap section forces plot."""
    data = [
        [x/10, instance.N(z), r'$N$'],
        [x/10, instance.M(z), r'$M$'],
        [x/10, instance.V(z), r'$V$']
    ]
    plot_data(ax1label=r'Section forces', ax1data=data,
              name='forc', **segments)


def stresses(instance, x, z, **segments):
    """Wrap stress plot."""
    data = [
        [x/10, instance.tau(z, unit='kPa'), r'$\tau$'],
        [x/10, instance.sig(z, unit='kPa'), r'$\sigma$']
    ]
    plot_data(ax1label=r'Stress (kPa)', ax1data=data,
              name='stress', **segments)


def stress_criteria(x, stress, **segments):
    """Wrap plot of stress and energy criteria."""
    data = [
        [x/10, stress, r'$\sigma/\sigma_\mathrm{c}$']
    ]
    plot_data(ax1label=r'Criteria', ax1data=data,
              name='crit', **segments)


def err_comp(da, Gdif, Ginc, mode=0):
    """Wrap energy release rate plot."""
    data = [
        [da/10, 1e3*Gdif[mode, :], r'$\mathcal{G}$'],
        [da/10, 1e3*Ginc[mode, :], r'$\bar{\mathcal{G}}$']
    ]
    plot_data(
        xlabel=r'Crack length $\Delta a$ (cm)',
        ax1label=r'Energy release rate (J/m$^2$)',
        ax1data=data, name='err', vlines=False)


def err_modes(da, G, kind='inc'):
    """Wrap energy release rate plot."""
    label = r'$\bar{\mathcal{G}}$' if kind == 'inc' else r'$\mathcal{G}$'
    data = [
        [da/10, 1e3*G[2, :], label + r'$_\mathrm{I\!I}$'],
        [da/10, 1e3*G[1, :], label + r'$_\mathrm{I}$'],
        [da/10, 1e3*G[0, :], label + r'$_\mathrm{I+I\!I}$']
    ]
    plot_data(
        xlabel=r'Crack length $a$ (cm)',
        ax1label=r'Energy release rate (J/m$^2$)',
        ax1data=data, name='modes', vlines=False)


def fea_disp(instance, x, z, fea):
    """Wrap dispalcements plot."""
    data = [
        [fea[:, 0]/10, -np.flipud(fea[:, 1]), r'FEA $u_0$'],
        [fea[:, 0]/10, np.flipud(fea[:, 2]), r'FEA $w_0$'],
        # [fea[:, 0]/10, -np.flipud(fea[:, 3]), r'FEA $u(z=-h/2)$'],
        # [fea[:, 0]/10, np.flipud(fea[:, 4]), r'FEA $w(z=-h/2)$'],
        [fea[:, 0]/10,
            np.flipud(np.rad2deg(fea[:, 5])), r'FEA $\psi$'],
        [x/10, instance.u(z, z0=0), r'$u_0$'],
        [x/10, -instance.w(z), r'$-w$'],
        [x/10, np.rad2deg(instance.psi(z)), r'$\psi$']
    ]
    plot_data(
        ax1label=r'Displacements (mm)', ax1data=data, name='fea_disp',
        labelpos=-50)


def fea_stress(instance, xb, zb, fea):
    """Wrap stress plot."""
    data = [
        [fea[:, 0]/10, 1e3*np.flipud(fea[:, 2]), r'FEA $\sigma_2$'],
        [fea[:, 0]/10, 1e3*np.flipud(fea[:, 3]), r'FEA $\tau_{12}$'],
        [xb/10, instance.tau(zb, unit='kPa'), r'$\tau$'],
        [xb/10, instance.sig(zb, unit='kPa'), r'$\sigma$']
    ]
    plot_data(ax1label=r'Stress (kPa)', ax1data=data, name='fea_stress',
              labelpos=-50)
