"""Plotting resources for the WEak Layer AntiCrack nucleation model."""
# pylint: disable=invalid-name,too-many-locals,too-many-branches
# pylint: disable=too-many-arguments,too-many-statements

# Standard library imports
import os
import colorsys

# Third party imports
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
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


class MidpointNormalize(mc.Normalize):
    """Colormap normalization to a specified midpoint. Default is 0."""

    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        """Inizialize normalization."""
        self.midpoint = midpoint
        mc.Normalize.__init__(self, vmin, vmax, clip)

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


def significant_digits(decimal):
    """
    Get the number of significant digits.

    Arguments
    ---------
    decimal : float
        Decimal number.

    Returns
    -------
    int
        Number of significant digits.
    """
    return -int(np.floor(np.log10(decimal)))


def tight_central_distribution(limit, samples=100, tightness=1.5):
    """
    Provide values within a given interval distributed tightly around 0.

    Parameters
    ----------
    limit : float
        Maximum and minimum of value range.
    samples : int, optional
        Number of values. Default is 100.
    tightness : int, optional
        Degree of value densification at center. 1.0 corresponds
        to equal spacing. Default is 1.5.

    Returns
    -------
    ndarray
        Array of values more tightly spaced around 0.
    """
    stop = limit**(1/tightness)
    levels = np.linspace(0, stop, num=int(samples/2), endpoint=True)**tightness
    return np.unique(np.hstack([-levels[::-1], levels]))


def adjust_lightness(color, amount=0.5):
    """
    Adjust color lightness.

    Arguments
    ----------
    color : str or tuple
        Matplotlib colorname, hex string, or RGB value tuple.
    amount : float, optional
        Amount of lightening: >1 lightens, <1 darkens. Default is 0.5.

    Returns
    -------
    tuple
        RGB color tuple.
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

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

    # Save figure
    save_plot(name='profile')

    # Reset plot styles
    plt.rcdefaults()

    # Clear Canvas
    plt.close()

# === DEFORMATION CONTOUR PLOT ================================================


def deformed(instance, xsl, xwl, z, phi, dz=2, scale=100,
             window=np.inf, pad=2, levels=300, aspect=2,
             field='principal', normalize=True, dark=False,
             filename='cont'):
    """
    Plot 2D deformed solution with displacement or stress fields.

    Arguments
    ---------
    instance : object
        Instance of layered class.
    xsl : ndarray
        Discretized slab x-coordinates (mm).
    xwl : ndarray
        Discretized weak-layer x-coordinates (mm).
    z : ndarray
        Solution vectors at positions x as columns of matrix z.
    phi : float
        Inclination (degrees). Counterclockwise positive.
    dz : float, optional
        Element size along z-axis (mm) for stress plot. Default is 2 mm.
    scale : int, optional
        Scaling factor for the visualization of displacements. Default
        is 100.
    window : int, optional
        Plot window (cm) around maximum vertical deflection. Default
        is inf (full view).
    pad : float, optional
        Padding around shown geometry. Default is 2.
    levels : int, optional
        Number of isolevels. Default is 300.
    aspect : int, optional
        Aspect ratio of the displayed geometry. 1 is true to scale.
        Default is 2.
    field : {'u', 'w', 'Sxx', 'Txz', 'Szz', 'principal'}, optional
        Field quantity for contour plot. Axial deformation 'u', vertical
        deflection 'w', axial normal stress 'Sxx', shear stress 'Txz',
        transverse normal stress 'Szz', or principal stresses 'principal'.
    normalize : bool, optional
        Toggle layerwise normalization of principal stresses to respective
        strength. Only available with field='principal'. Default is True.
    dark : bool, optional
        Toggle display on dark figure background. Default is False.

    Raises
    ------
    ValueError
        If invalid stress or displacement field is requested.
    """
    # Plot Setup
    plt.rcdefaults()
    plt.rc('font', family='serif', size=10)
    plt.rc('mathtext', fontset='cm')

    # Set dark figure background if requested
    if dark:
        plt.style.use('dark_background')
        fig = plt.figure()
        ax = plt.gca()
        fig.set_facecolor('#282c34')
        ax.set_facecolor('white')

    # Calculate top-to-bottom vertical positions (mm) in beam coordinate system
    zi = instance.get_zmesh(dz=dz)[:, 0]
    h = instance.h

    # Compute slab displacements on grid (cm)
    Usl = np.vstack([instance.u(z, z0=z0, unit='cm') for z0 in zi])
    Wsl = np.vstack([instance.w(z, unit='cm') for _ in zi])

    # Put coordinate origin at horizontal center
    if instance.system in ['skier', 'skiers']:
        xsl = xsl - max(xsl)/2
        xwl = xwl - max(xwl)/2

    # Compute slab grid coordinates with vertical origin at top surface (cm)
    Xsl, Zsl = np.meshgrid(1e-1*(xsl), 1e-1*(zi + h/2))

    # Get x-coordinate of maximum deflection w (cm) and derive plot limits
    xfocus = xsl[np.max(np.argmax(Wsl, axis=1))]/10
    xmax = np.min([np.max([Xsl, Xsl+scale*Usl]) + pad, xfocus + window/2])
    xmin = np.max([np.min([Xsl, Xsl+scale*Usl]) - pad, xfocus - window/2])

    # Scale shown weak-layer thickness with to max deflection and add padding
    zmax = np.max(Zsl + scale*Wsl) + pad
    zmin = np.min(Zsl) - pad

    # Compute weak-layer grid coordinates (cm)
    Xwl, Zwl = np.meshgrid(1e-1*xwl, [1e-1*(zi[-1] + h/2), zmax])

    # Assemble weak-layer displacement field (top and bottom)
    Uwl = np.row_stack([Usl[-1, :], np.zeros(xwl.shape[0])])
    Wwl = np.row_stack([Wsl[-1, :], np.zeros(xwl.shape[0])])

    # Compute stress or displacement fields
    match field:
        # Horizontal displacements (um)
        case 'u':
            slab = 1e4*Usl
            weak = 1e4*Usl[-1, :]
            label = r'$u$ ($\mu$m)'
        # Vertical deflection (um)
        case 'w':
            slab = 1e4*Wsl
            weak = 1e4*Wsl[-1, :]
            label = r'$w$ ($\mu$m)'
        # Axial normal stresses (kPa)
        case 'Sxx':
            slab = instance.Sxx(z, phi, dz=dz, unit='kPa')
            weak = np.zeros(xwl.shape[0])
            label = r'$\sigma_{xx}$ (kPa)'
        # Shear stresses (kPa)
        case 'Txz':
            slab = instance.Txz(z, phi, dz=dz, unit='kPa')
            weak = instance.get_weaklayer_shearstress(
                x=xwl, z=z, unit='kPa')[1]
            label = r'$\tau_{xz}$ (kPa)'
        # Transverse normal stresses (kPa)
        case 'Szz':
            slab = instance.Szz(z, phi, dz=dz, unit='kPa')
            weak = instance.get_weaklayer_normalstress(
                x=xwl, z=z, unit='kPa')[1]
            label = r'$\sigma_{zz}$ (kPa)'
        # Principal stresses
        case 'principal':
            slab = instance.principal_stress_slab(
                z, phi, dz=dz, val='max', unit='kPa', normalize=normalize)
            weak = instance.principal_stress_weaklayer(
                z, val='min', unit='kPa', normalize=normalize)
            if normalize:
                label=(r'$\sigma_\mathrm{I}/\sigma_+$ (slab),  '
                       r'$\sigma_\mathrm{I\!I\!I}/\sigma_-$ (weak layer)')
            else:
                label=(r'$\sigma_\mathrm{I}$ (kPa, slab),  '
                       r'$\sigma_\mathrm{I\!I\!I}$ (kPa, weak layer)')
        case _:
            raise ValueError(
                f"Invalid input '{field}' for field. Valid options are "
                "'u', 'w', 'Sxx', 'Txz', 'Szz', or 'principal'")

    # Complement label
    label += r'  $\longrightarrow$'

    # Assemble weak-layer output on grid
    weak = np.row_stack([weak, weak])

    # Normalize colormap
    absmax = np.nanmax(np.abs([slab.min(), slab.max(), weak.min(), weak.max()]))
    clim = np.round(absmax, significant_digits(absmax))
    levels = np.linspace(-clim, clim, num=levels+1, endpoint=True)
    # nanmax = np.nanmax([slab.max(), weak.max()])
    # nanmin = np.nanmin([slab.min(), weak.min()])
    # norm = MidpointNormalize(vmin=nanmin, vmax=nanmax)

    # Plot baseline
    plt.axhline(zmax, color='k', linewidth=1)

    # Plot outlines of the undeformed and deformed slab
    plt.plot(outline(Xsl), outline(Zsl), 'k--', alpha=0.3, linewidth=1)
    plt.plot(outline(Xsl + scale*Usl),
             outline(Zsl + scale*Wsl),
             'k', linewidth=1)

    # Plot deformed weak-layer outline
    if instance.system in ['-pst', 'pst-', '-vpst', 'vpst-']:
        nanmask = np.isfinite(xwl)
        plt.plot(outline(Xwl[:, nanmask] + scale*Uwl[:, nanmask]),
                 outline(Zwl[:, nanmask] + scale*Wwl[:, nanmask]),
                 'k', linewidth=1)

    # Colormap
    cmap = plt.cm.RdBu_r
    cmap.set_over(adjust_lightness(cmap(1.0), 0.9))
    cmap.set_under(adjust_lightness(cmap(0.0), 0.9))

    # Plot fields
    plt.contourf(Xsl + scale*Usl, Zsl + scale*Wsl, slab,
                 levels=levels, # norm=norm,
                 cmap=cmap, extend='both')
    plt.contourf(Xwl + scale*Uwl, Zwl + scale*Wwl, weak,
                 levels=levels, # norm=norm,
                 cmap=cmap, extend='both')

    # Plot setup
    plt.axis('scaled')
    plt.xlim([xmin, xmax])
    plt.ylim([zmin, zmax])
    plt.gca().set_aspect(aspect)
    plt.gca().invert_yaxis()
    plt.gca().use_sticky_edges = False

    # Plot labels
    plt.gca().set_xlabel(r'lateral position $x$ (cm) $\longrightarrow$')
    plt.gca().set_ylabel('depth below surface\n' + r'$\longleftarrow $ $d$ (cm)')
    plt.title(fr'${scale}\!\times\!$ scaled deformations (cm)', size=10)

    # Show colorbar
    ticks = np.linspace(levels[0], levels[-1], num=11, endpoint=True)
    plt.colorbar(orientation='horizontal', ticks=ticks,
                 label=label, aspect=35)

    # Save figure
    save_plot(name=filename)

    # Clear Canvas
    plt.close()

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
    save_plot(name)

    # Clear canvas
    plt.close()

    # Reset plot styles
    plt.rcdefaults()


# === PLOT WRAPPERS ===========================================================


def displacements(instance, x, z, i='', **segments):
    """Wrap for dispalcements plot."""
    data = [
        [x/10, instance.u(z, z0=0, unit='mm'), r'$u_0\ (\mathrm{mm})$'],
        [x/10, -instance.w(z, unit='mm'), r'$-w\ (\mathrm{mm})$'],
        [x/10, instance.psi(z, unit='degrees'), r'$\psi\ (^\circ)$ '],
    ]
    plot_data(ax1label=r'Displacements', ax1data=data,
              name='disp' + str(i), **segments)


def section_forces(instance, x, z, i='', **segments):
    """Wrap section forces plot."""
    data = [
        [x/10, instance.N(z), r'$N$'],
        [x/10, instance.M(z), r'$M$'],
        [x/10, instance.V(z), r'$V$']
    ]
    plot_data(ax1label=r'Section forces', ax1data=data,
              name='forc' + str(i), **segments)


def stresses(instance, x, z, i='', **segments):
    """Wrap stress plot."""
    data = [
        [x/10, instance.tau(z, unit='kPa'), r'$\tau$'],
        [x/10, instance.sig(z, unit='kPa'), r'$\sigma$']
    ]
    plot_data(ax1label=r'Stress (kPa)', ax1data=data,
              name='stress' + str(i), **segments)


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


# === SAVE FUNCTION ===========================================================

def save_plot(name):
    """
    Show or save plot depending on interpreter

    Arguments
    ---------
    name : string
        Name for the figure.
    """
    filename = name + '.png'
    # Show figure if on jupyter notebook
    if isnotebook():
        plt.show()
    # Save figure if on terminal
    else:
        # Make directory if not yet existing
        if not os.path.isdir(os.path.join(os.getcwd(), 'plots')):
            os.mkdir('plots')
        plt.savefig('plots/' + filename, bbox_inches='tight')
    return
