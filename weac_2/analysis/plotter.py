# Standard library imports
import colorsys
import os
from typing import List, Literal, Optional

# Third party imports
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

from weac_2.analysis.analyzer import Analyzer

# Module imports
from weac_2.core.scenario import Scenario
from weac_2.core.system_model import SystemModel
from weac_2.utils import isnotebook

LABELSTYLE = {
    "backgroundcolor": "w",
    "horizontalalignment": "center",
    "verticalalignment": "center",
}

COLORS = np.array(
    [  # TUD color palette
        ["#DCDCDC", "#B5B5B5", "#898989", "#535353"],  # gray
        ["#5D85C3", "#005AA9", "#004E8A", "#243572"],  # blue
        ["#009CDA", "#0083CC", "#00689D", "#004E73"],  # ocean
        ["#50B695", "#009D81", "#008877", "#00715E"],  # teal
        ["#AFCC50", "#99C000", "#7FAB16", "#6A8B22"],  # green
        ["#DDDF48", "#C9D400", "#B1BD00", "#99A604"],  # lime
        ["#FFE05C", "#FDCA00", "#D7AC00", "#AE8E00"],  # yellow
        ["#F8BA3C", "#F5A300", "#D28700", "#BE6F00"],  # sand
        ["#EE7A34", "#EC6500", "#CC4C03", "#A94913"],  # orange
        ["#E9503E", "#E6001A", "#B90F22", "#961C26"],  # red
        ["#C9308E", "#A60084", "#951169", "#732054"],  # magenta
        ["#804597", "#721085", "#611C73", "#4C226A"],  # purple
    ]
)


def _outline(grid):
    """Extract _outline values of a 2D array (matrix, grid)."""
    top = grid[0, :-1]
    right = grid[:-1, -1]
    bot = grid[-1, :0:-1]
    left = grid[::-1, 0]

    return np.hstack([top, right, bot, left])


def _significant_digits(decimal):
    """Return the number of significant digits for a given decimal."""
    return -int(np.floor(np.log10(decimal)))


def _tight_central_distribution(limit, samples=100, tightness=1.5):
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
    stop = limit ** (1 / tightness)
    levels = np.linspace(0, stop, num=int(samples / 2), endpoint=True) ** tightness
    return np.unique(np.hstack([-levels[::-1], levels]))


def _adjust_lightness(color, amount=0.5):
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


class MidpointNormalize(mc.Normalize):
    """Colormap normalization to a specified midpoint. Default is 0."""

    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        """Initialize normalization."""
        self.midpoint = midpoint
        mc.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """Apply normalization."""
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Plotter:
    """
    Modern plotting class for WEAC simulations with support for multiple system comparisons.

    This class provides comprehensive visualization capabilities for weak layer anticrack
    nucleation simulations, including single system analysis and multi-system comparisons.

    Features:
    - Single and multi-system plotting
    - System override functionality for selective plotting
    - Comprehensive dashboard creation
    - Modern matplotlib styling
    - Jupyter notebook integration
    - Automatic plot directory management
    """

    def __init__(
        self,
        system: Optional[SystemModel] = None,
        systems: Optional[List[SystemModel]] = None,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        plot_dir: str = "plots",
    ):
        """
        Initialize the plotter.

        Parameters
        ----------
        system : SystemModel, optional
            Single system model for analysis
        systems : List[SystemModel], optional
            List of system models for comparison
        labels : List[str], optional
            Labels for each system in plots
        colors : List[str], optional
            Colors for each system in plots
        plot_dir : str, default "plots"
            Directory to save plots
        """
        # Handle system input
        if system is not None and systems is not None:
            raise ValueError("Provide either 'system' or 'systems', not both")
        elif system is not None:
            self.systems = [system]
        elif systems is not None:
            self.systems = systems
        else:
            raise ValueError("Must provide either 'system' or 'systems'")

        self.n_systems = len(self.systems)

        # Set up labels
        if labels is None:
            self.labels = [f"System {i + 1}" for i in range(self.n_systems)]
        else:
            if len(labels) != self.n_systems:
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match number of systems ({self.n_systems})"
                )
            self.labels = labels

        self.colors = COLORS

        # Set up plot directory
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        # Set up matplotlib style
        self._setup_matplotlib_style()

        # Cache analyzers for performance
        self._analyzers = {}

    def _setup_matplotlib_style(self):
        """Set up modern matplotlib styling."""
        plt.style.use("default")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "figure.dpi": 100,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "lines.linewidth": 2,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "axes.axisbelow": True,
            }
        )

    def _get_analyzer(self, system: SystemModel) -> Analyzer:
        """Get cached analyzer for a system."""
        system_id = id(system)
        if system_id not in self._analyzers:
            self._analyzers[system_id] = Analyzer(system_model=system)
        return self._analyzers[system_id]

    def _get_systems_to_plot(
        self,
        system_model: Optional[SystemModel] = None,
        system_models: Optional[List[SystemModel]] = None,
    ) -> List[SystemModel]:
        """Determine which systems to plot based on override parameters."""
        if system_model is not None and system_models is not None:
            raise ValueError(
                "Provide either 'system_model' or 'system_models', not both"
            )
        elif system_model is not None:
            return [system_model]
        elif system_models is not None:
            return system_models
        else:
            return self.systems

    def _save_figure(self, filename: str, fig: Optional[plt.Figure] = None):
        """Save figure with proper formatting."""
        if fig is None:
            fig = plt.gcf()

        filepath = os.path.join(self.plot_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")

        if not isnotebook():
            plt.close(fig)

    def plot_slab_profile(
        self,
        system_model: Optional[SystemModel] = None,
        system_models: Optional[List[SystemModel]] = None,
        filename: Optional[str] = None,
    ):
        """
        Plot slab layer profiles for comparison.

        Parameters
        ----------
        system_model : SystemModel, optional
            Single system to plot (overrides default)
        system_models : List[SystemModel], optional
            Multiple systems to plot (overrides default)
        filename : str, optional
            Filename for saving plot

        Returns
        -------
        matplotlib.axes.Axes
            The generated plot axes.
        """
        systems_to_plot = self._get_systems_to_plot(system_model, system_models)
        labels, colors = self.labels, self.colors

        # Plot Setup
        plt.rcdefaults()
        plt.rc("font", family="serif", size=10)
        plt.rc("mathtext", fontset="cm")

        fig = plt.figure(figsize=(4, 7))
        ax1 = fig.gca()

        # Plot 1: Layer thickness and density
        max_height = 0
        for system in systems_to_plot:
            total_height = system.slab.H + system.weak_layer.h
            max_height = max(max_height, total_height)

        for i, (system, label, color) in enumerate(
            zip(systems_to_plot, labels, colors)
        ):
            # Plot weak layer
            wl_y = [-system.weak_layer.h, 0]
            wl_x = [system.weak_layer.rho, system.weak_layer.rho]
            ax1.fill_betweenx(wl_y, 0, wl_x, color="red", alpha=0.8, hatch="///")

            # Plot slab layers
            x_coords = []
            y_coords = []
            current_height = 0

            # As slab.layers is top-down
            for layer in reversed(system.slab.layers):
                x_coords.extend([layer.rho, layer.rho])
                y_coords.extend([current_height, current_height + layer.h])
                current_height += layer.h

            ax1.fill_betweenx(
                y_coords, 0, x_coords, color=color, alpha=0.7, label=label
            )

        # Set axis labels
        ax1.set_xlabel(r"$\longleftarrow$ Density $\rho$ (kg/m$^3$)")
        ax1.set_ylabel(r"Height above weak layer (mm) $\longrightarrow$")

        ax1.set_title("Slab Density Profile")

        # Create custom legend
        from matplotlib.patches import Patch

        handles, slab_labels = ax1.get_legend_handles_labels()
        weak_layer_patch = Patch(
            facecolor="red", alpha=0.8, hatch="///", label="Weak Layer"
        )
        ax1.legend(
            handles=[weak_layer_patch] + handles, labels=["Weak Layer"] + slab_labels
        )

        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(500, 0)
        ax1.set_ylim(-system.weak_layer.h, max_height)

        if filename:
            self._save_figure(filename, fig)

        # Reset plot styles
        plt.rcdefaults()

    def plot_section_forces(
        self,
        system_model: Optional[SystemModel] = None,
        system_models: Optional[List[SystemModel]] = None,
        filename: Optional[str] = None,
    ):
        """
        Plot section forces (N, M, V) for comparison.

        Parameters
        ----------
        system_model : SystemModel, optional
            Single system to plot (overrides default)
        system_models : List[SystemModel], optional
            Multiple systems to plot (overrides default)
        filename : str, optional
            Filename for saving plot
        """
        systems_to_plot = self._get_systems_to_plot(system_model, system_models)
        labels, colors = self.labels, self.colors

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        for system, label, color in zip(systems_to_plot, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            fq = system.fq

            # Convert x to meters for plotting
            x_m = x / 1000

            # Plot axial force N
            N = fq.N(z)
            axes[0].plot(x_m, N, color=color, label=label, linewidth=2)

            # Plot bending moment M
            M = fq.M(z)
            axes[1].plot(x_m, M, color=color, label=label, linewidth=2)

            # Plot shear force V
            V = fq.V(z)
            axes[2].plot(x_m, V, color=color, label=label, linewidth=2)

        # Formatting
        axes[0].set_ylabel("N (N)")
        axes[0].set_title("Axial Force")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("M (Nmm)")
        axes[1].set_title("Bending Moment")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel("Distance (m)")
        axes[2].set_ylabel("V (N)")
        axes[2].set_title("Shear Force")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            self._save_figure(filename, fig)

        return fig

    def plot_energy_release_rates(
        self,
        system_model: Optional[SystemModel] = None,
        system_models: Optional[List[SystemModel]] = None,
        filename: Optional[str] = None,
    ):
        """
        Plot energy release rates (G_I, G_II) for comparison.

        Parameters
        ----------
        system_model : SystemModel, optional
            Single system to plot (overrides default)
        system_models : List[SystemModel], optional
            Multiple systems to plot (overrides default)
        filename : str, optional
            Filename for saving plot
        """
        systems_to_plot = self._get_systems_to_plot(system_model, system_models)
        labels, colors = self.labels, self.colors

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        for system, label, color in zip(systems_to_plot, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            fq = system.fq

            # Convert x to meters for plotting
            x_m = x / 1000

            # Plot Mode I energy release rate
            G_I = fq.Gi(z, unit="kJ/m^2")
            axes[0].plot(x_m, G_I, color=color, label=label, linewidth=2)

            # Plot Mode II energy release rate
            G_II = fq.Gii(z, unit="kJ/m^2")
            axes[1].plot(x_m, G_II, color=color, label=label, linewidth=2)

        # Formatting
        axes[0].set_ylabel("G_I (kJ/m²)")
        axes[0].set_title("Mode I Energy Release Rate")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Distance (m)")
        axes[1].set_ylabel("G_II (kJ/m²)")
        axes[1].set_title("Mode II Energy Release Rate")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            self._save_figure(filename, fig)

        return fig

    def plot_deformed(
        self,
        xsl: np.ndarray,
        xwl: np.ndarray,
        z: np.ndarray,
        analyzer: Analyzer,
        dz: int = 2,
        scale: int = 100,
        window: int = np.inf,
        pad: int = 2,
        levels: int = 300,
        aspect: int = 2,
        field: Literal["w", "u", "principal", "Sxx", "Txz", "Szz"] = "w",
        normalize: bool = True,
        filename: Optional[str] = None,
    ):
        """
        Plot deformed slab with field contours.

        Parameters
        ----------
        field : str, default 'w'
            Field to plot ('w', 'u', 'principal', 'sigma', 'tau')
        system_model : SystemModel, optional
            System to plot (uses first system if not specified)
        filename : str, optional
            Filename for saving plot
        """
        # Plot Setup
        plt.rcdefaults()
        plt.rc("font", family="serif", size=10)
        plt.rc("mathtext", fontset="cm")

        zi = analyzer.get_zmesh(dz=dz)["z"]
        H = analyzer.sm.slab.H
        phi = analyzer.sm.scenario.phi
        system_type = analyzer.sm.scenario.system_type
        fq = analyzer.sm.fq

        # Compute slab displacements on grid (cm)
        Usl = np.vstack([fq.u(z, h0=h0, unit="cm") for h0 in zi])
        Wsl = np.vstack([fq.w(z, unit="cm") for _ in zi])

        # Put coordinate origin at horizontal center
        if system_type in ["skier", "skiers"]:
            xsl = xsl - max(xsl) / 2
            xwl = xwl - max(xwl) / 2

        # Compute slab grid coordinates with vertical origin at top surface (cm)
        Xsl, Zsl = np.meshgrid(1e-1 * (xsl), 1e-1 * (zi + H / 2))

        # Get x-coordinate of maximum deflection w (cm) and derive plot limits
        xfocus = xsl[np.max(np.argmax(Wsl, axis=1))] / 10
        xmax = np.min([np.max([Xsl, Xsl + scale * Usl]) + pad, xfocus + window / 2])
        xmin = np.max([np.min([Xsl, Xsl + scale * Usl]) - pad, xfocus - window / 2])

        # Scale shown weak-layer thickness with to max deflection and add padding
        if analyzer.sm.config.touchdown:
            zmax = (
                np.max(Zsl)
                + (analyzer.sm.weak_layer.h * 1e-1 * scale)
                - (analyzer.sm.scenario.crack_h * 1e-1 * scale)
            )
            zmax = min(zmax, np.max(Zsl + scale * Wsl))
        else:
            zmax = np.max(Zsl + scale * Wsl)
        zmin = np.min(Zsl) - pad

        # Compute weak-layer grid coordinates (cm)
        Xwl, Zwl = np.meshgrid(1e-1 * xwl, [1e-1 * (zi[-1] + H / 2), zmax])

        # Assemble weak-layer displacement field (top and bottom)
        Uwl = np.vstack([Usl[-1, :], np.zeros(xwl.shape[0])])
        Wwl = np.vstack([Wsl[-1, :], np.zeros(xwl.shape[0])])

        # Compute stress or displacement fields
        match field:
            # Horizontal displacements (um)
            case "u":
                slab = 1e4 * Usl
                weak = 1e4 * Usl[-1, :]
                label = r"$u$ ($\mu$m)"
            # Vertical deflection (um)
            case "w":
                slab = 1e4 * Wsl
                weak = 1e4 * Wsl[-1, :]
                label = r"$w$ ($\mu$m)"
            # Axial normal stresses (kPa)
            case "Sxx":
                slab = analyzer.Sxx(z, phi, dz=dz, unit="kPa")
                weak = np.zeros(xwl.shape[0])
                label = r"$\sigma_{xx}$ (kPa)"
            # Shear stresses (kPa)
            case "Txz":
                slab = analyzer.Txz(z, phi, dz=dz, unit="kPa")
                weak = analyzer.weaklayer_shearstress(x=xwl, z=z, unit="kPa")[1]
                label = r"$\tau_{xz}$ (kPa)"
            # Transverse normal stresses (kPa)
            case "Szz":
                slab = analyzer.Szz(z, phi, dz=dz, unit="kPa")
                weak = analyzer.weaklayer_normalstress(x=xwl, z=z, unit="kPa")[1]
                label = r"$\sigma_{zz}$ (kPa)"
            # Principal stresses
            case "principal":
                slab = analyzer.principal_stress_slab(
                    z, phi, dz=dz, val="max", unit="kPa", normalize=normalize
                )
                weak = analyzer.principal_stress_weaklayer(
                    z, val="min", unit="kPa", normalize=normalize
                )
                if normalize:
                    label = (
                        r"$\sigma_\mathrm{I}/\sigma_+$ (slab),  "
                        r"$\sigma_\mathrm{I\!I\!I}/\sigma_-$ (weak layer)"
                    )
                else:
                    label = (
                        r"$\sigma_\mathrm{I}$ (kPa, slab),  "
                        r"$\sigma_\mathrm{I\!I\!I}$ (kPa, weak layer)"
                    )
            case _:
                raise ValueError(
                    f"Invalid input '{field}' for field. Valid options are "
                    "'u', 'w', 'Sxx', 'Txz', 'Szz', or 'principal'"
                )

        # Complement label
        label += r"  $\longrightarrow$"

        # Assemble weak-layer output on grid
        weak = np.vstack([weak, weak])

        # Normalize colormap
        absmax = np.nanmax(np.abs([slab.min(), slab.max(), weak.min(), weak.max()]))
        clim = np.round(absmax, _significant_digits(absmax))
        levels = np.linspace(-clim, clim, num=levels + 1, endpoint=True)
        # nanmax = np.nanmax([slab.max(), weak.max()])
        # nanmin = np.nanmin([slab.min(), weak.min()])
        # norm = MidpointNormalize(vmin=nanmin, vmax=nanmax)

        # Plot baseline
        plt.axhline(zmax, color="k", linewidth=1)

        # Plot outlines of the undeformed and deformed slab
        plt.plot(_outline(Xsl), _outline(Zsl), "k--", alpha=0.3, linewidth=1)
        plt.plot(
            _outline(Xsl + scale * Usl), _outline(Zsl + scale * Wsl), "k", linewidth=1
        )

        # Plot deformed weak-layer _outline
        if system_type in ["-pst", "pst-", "-vpst", "vpst-"]:
            nanmask = np.isfinite(xwl)
            plt.plot(
                _outline(Xwl[:, nanmask] + scale * Uwl[:, nanmask]),
                _outline(Zwl[:, nanmask] + scale * Wwl[:, nanmask]),
                "k",
                linewidth=1,
            )

        # Colormap
        cmap = plt.cm.RdBu_r
        cmap.set_over(_adjust_lightness(cmap(1.0), 0.9))
        cmap.set_under(_adjust_lightness(cmap(0.0), 0.9))

        # Plot fields
        plt.contourf(
            Xsl + scale * Usl,
            Zsl + scale * Wsl,
            slab,
            levels=levels,  # norm=norm,
            cmap=cmap,
            extend="both",
        )
        plt.contourf(
            Xwl + scale * Uwl,
            Zwl + scale * Wwl,
            weak,
            levels=levels,  # norm=norm,
            cmap=cmap,
            extend="both",
        )

        # Plot setup
        plt.axis("scaled")
        plt.xlim([xmin, xmax])
        plt.ylim([zmin, zmax])
        plt.gca().set_aspect(aspect)
        plt.gca().invert_yaxis()
        plt.gca().use_sticky_edges = False

        # Plot labels
        plt.gca().set_xlabel(r"lateral position $x$ (cm) $\longrightarrow$")
        plt.gca().set_ylabel("depth below surface\n" + r"$\longleftarrow $ $d$ (cm)")
        plt.title(rf"${scale}\!\times\!$ scaled deformations (cm)", size=10)

        # Show colorbar
        ticks = np.linspace(levels[0], levels[-1], num=11, endpoint=True)
        plt.colorbar(orientation="horizontal", ticks=ticks, label=label, aspect=35)

        # Save figure
        self._save_figure(filename)

        # Reset plot styles
        plt.rcdefaults()

    def plot_stress_envelope(
        self, system_model: Optional[SystemModel] = None, filename: Optional[str] = None
    ):
        """
        Plot stress envelope in τ-σ space.

        Parameters
        ----------
        system_model : SystemModel, optional
            System to plot (uses first system if not specified)
        filename : str, optional
            Filename for saving plot
        """
        if system_model is None:
            system_model = self.systems[0]

        analyzer = self._get_analyzer(system_model)
        x, z, _ = analyzer.rasterize_solution()
        fq = system_model.fq

        # Calculate stresses
        sigma = fq.sig(z, unit="kPa")
        tau = fq.tau(z, unit="kPa")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot stress path
        ax.plot(sigma, tau, "b-", linewidth=2, label="Stress Path")
        ax.scatter(
            sigma[0], tau[0], color="green", s=100, marker="o", label="Start", zorder=5
        )
        ax.scatter(
            sigma[-1], tau[-1], color="red", s=100, marker="s", label="End", zorder=5
        )

        # Add failure envelope (simplified Mohr-Coulomb)
        sigma_range = np.linspace(min(sigma.min(), 0), sigma.max() * 1.1, 100)

        # Typical values for snow (these could be made configurable)
        cohesion = 2.0  # kPa
        friction_angle = 30  # degrees
        friction_coeff = np.tan(np.deg2rad(friction_angle))

        tau_envelope = cohesion + friction_coeff * np.abs(sigma_range)
        ax.plot(sigma_range, tau_envelope, "r--", linewidth=2, label="Failure Envelope")
        ax.plot(sigma_range, -tau_envelope, "r--", linewidth=2)

        # Formatting
        ax.set_xlabel("Normal Stress σ (kPa)")
        ax.set_ylabel("Shear Stress τ (kPa)")
        ax.set_title("Weak Layer Stress Envelope")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

        plt.tight_layout()

        if filename:
            self._save_figure(filename, fig)

        return fig

    def create_comparison_dashboard(
        self,
        system_models: Optional[List[SystemModel]] = None,
        filename: Optional[str] = None,
    ):
        """
        Create a comprehensive comparison dashboard.

        Parameters
        ----------
        system_models : List[SystemModel], optional
            Systems to include in dashboard (uses all if not specified)
        filename : str, optional
            Filename for saving plot
        """
        if system_models is None:
            system_models = self.systems

        labels, colors = self.labels, self.colors

        fig = plt.figure(figsize=(20, 16))

        # Create subplot grid
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Slab profiles
        ax1 = fig.add_subplot(gs[0, 0])
        for system, label, color in zip(system_models, labels, colors):
            slab = system.slab
            z_positions = np.concatenate(
                [[0], np.cumsum([layer.h for layer in slab.layers])]
            )
            densities = [layer.rho for layer in slab.layers]

            for j, (z_start, z_end, rho) in enumerate(
                zip(z_positions[:-1], z_positions[1:], densities)
            ):
                ax1.barh(
                    z_start,
                    rho,
                    height=z_end - z_start,
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    label=label if j == 0 else "",
                )

        ax1.set_xlabel("Density (kg/m³)")
        ax1.set_ylabel("Height (mm)")
        ax1.set_title("Slab Profiles")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Vertical displacement
        ax2 = fig.add_subplot(gs[0, 1])
        for system, label, color in zip(system_models, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            w = system.fq.w(z, unit="mm")
            ax2.plot(x / 1000, w, color=color, label=label, linewidth=2)

        ax2.set_xlabel("Distance (m)")
        ax2.set_ylabel("w (mm)")
        ax2.set_title("Vertical Displacement")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Normal stress
        ax3 = fig.add_subplot(gs[0, 2])
        for system, label, color in zip(system_models, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            sigma = system.fq.sig(z, unit="kPa")
            ax3.plot(x / 1000, sigma, color=color, label=label, linewidth=2)

        ax3.set_xlabel("Distance (m)")
        ax3.set_ylabel("σ (kPa)")
        ax3.set_title("Normal Stress")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Shear stress
        ax4 = fig.add_subplot(gs[1, 0])
        for system, label, color in zip(system_models, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            tau = system.fq.tau(z, unit="kPa")
            ax4.plot(x / 1000, tau, color=color, label=label, linewidth=2)

        ax4.set_xlabel("Distance (m)")
        ax4.set_ylabel("τ (kPa)")
        ax4.set_title("Shear Stress")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Bending moment
        ax5 = fig.add_subplot(gs[1, 1])
        for system, label, color in zip(system_models, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            M = system.fq.M(z)
            ax5.plot(x / 1000, M, color=color, label=label, linewidth=2)

        ax5.set_xlabel("Distance (m)")
        ax5.set_ylabel("M (Nmm)")
        ax5.set_title("Bending Moment")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Energy release rates
        ax6 = fig.add_subplot(gs[1, 2])
        for system, label, color in zip(system_models, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            G_I = system.fq.Gi(z, unit="kJ/m^2")
            G_II = system.fq.Gii(z, unit="kJ/m^2")
            ax6.plot(x / 1000, G_I + G_II, color=color, label=label, linewidth=2)

        ax6.set_xlabel("Distance (m)")
        ax6.set_ylabel("G_total (kJ/m²)")
        ax6.set_title("Total Energy Release Rate")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7-9. System information table
        ax7 = fig.add_subplot(gs[2:, :])
        ax7.axis("off")

        # Create system information table
        table_data = []
        headers = [
            "System",
            "Slope (°)",
            "Slab H (mm)",
            "WL h (mm)",
            "WL ρ (kg/m³)",
            "Max |w| (mm)",
            "Max |τ| (kPa)",
        ]

        for i, (system, label) in enumerate(zip(system_models, labels)):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()

            max_w = np.max(np.abs(system.fq.w(z, unit="mm")))
            max_tau = np.max(np.abs(system.fq.tau(z, unit="kPa")))

            row = [
                label,
                f"{system.scenario.phi:.1f}",
                f"{system.slab.H:.0f}",
                f"{system.weak_layer.h:.0f}",
                f"{system.weak_layer.rho:.0f}",
                f"{max_w:.3f}",
                f"{max_tau:.3f}",
            ]
            table_data.append(row)

        table = ax7.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colColours=["lightgray"] * len(headers),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        ax7.set_title("System Comparison Summary", fontsize=16, pad=20)

        plt.suptitle("WEAC Simulation Comparison Dashboard", fontsize=18, y=0.98)

        if filename:
            self._save_figure(filename, fig)

        return fig

    # === PLOT WRAPPERS ===========================================================

    def plot_displacements(
        self, analyzer: Analyzer, x: np.ndarray, z: np.ndarray, i: int = 0
    ):
        """Wrap for displacements plot."""
        data = [
            [x / 10, analyzer.sm.fq.u(z, unit="mm"), r"$u_0\ (\mathrm{mm})$"],
            [x / 10, -analyzer.sm.fq.w(z, unit="mm"), r"$-w\ (\mathrm{mm})$"],
            [x / 10, analyzer.sm.fq.psi(z, unit="deg"), r"$\psi\ (^\circ)$ "],
        ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            ax1label=r"Displacements",
            ax1data=data,
            name="disp" + str(i),
        )

    def plot_stresses(
        self, analyzer: Analyzer, x: np.ndarray, z: np.ndarray, i: int = 0
    ):
        """Wrap stress plot."""
        data = [
            [x / 10, analyzer.sm.fq.tau(z, unit="kPa"), r"$\tau$"],
            [x / 10, analyzer.sm.fq.sig(z, unit="kPa"), r"$\sigma$"],
        ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            ax1label=r"Stress (kPa)",
            ax1data=data,
            name="stress" + str(i),
        )

    def plot_stress_criteria(
        self, analyzer: Analyzer, x: np.ndarray, stress: np.ndarray
    ):
        """Wrap plot of stress and energy criteria."""
        data = [[x / 10, stress, r"$\sigma/\sigma_\mathrm{c}$"]]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            ax1label=r"Criteria",
            ax1data=data,
            name="crit",
        )

    def plot_ERR_comp(
        self,
        analyzer: Analyzer,
        da: np.ndarray,
        Gdif: np.ndarray,
        Ginc: np.ndarray,
        mode: int = 0,
    ):
        """Wrap energy release rate plot."""
        data = [
            [da / 10, 1e3 * Gdif[mode, :], r"$\mathcal{G}$"],
            [da / 10, 1e3 * Ginc[mode, :], r"$\bar{\mathcal{G}}$"],
        ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            xlabel=r"Crack length $\Delta a$ (cm)",
            ax1label=r"Energy release rate (J/m$^2$)",
            ax1data=data,
            name="err",
            vlines=False,
        )

    def plot_ERR_modes(
        self, analyzer: Analyzer, da: np.ndarray, G: np.ndarray, kind: str = "inc"
    ):
        """Wrap energy release rate plot."""
        label = r"$\bar{\mathcal{G}}$" if kind == "inc" else r"$\mathcal{G}$"
        data = [
            [da / 10, 1e3 * G[2, :], label + r"$_\mathrm{I\!I}$"],
            [da / 10, 1e3 * G[1, :], label + r"$_\mathrm{I}$"],
            [da / 10, 1e3 * G[0, :], label + r"$_\mathrm{I+I\!I}$"],
        ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            xlabel=r"Crack length $a$ (cm)",
            ax1label=r"Energy release rate (J/m$^2$)",
            ax1data=data,
            name="modes",
            vlines=False,
        )

    def plot_fea_disp(
        self, analyzer: Analyzer, x: np.ndarray, z: np.ndarray, fea: np.ndarray
    ):
        """Wrap displacements plot."""
        data = [
            [fea[:, 0] / 10, -np.flipud(fea[:, 1]), r"FEA $u_0$"],
            [fea[:, 0] / 10, np.flipud(fea[:, 2]), r"FEA $w_0$"],
            # [fea[:, 0]/10, -np.flipud(fea[:, 3]), r'FEA $u(z=-h/2)$'],
            # [fea[:, 0]/10, np.flipud(fea[:, 4]), r'FEA $w(z=-h/2)$'],
            [fea[:, 0] / 10, np.flipud(np.rad2deg(fea[:, 5])), r"FEA $\psi$"],
            [x / 10, analyzer.sm.fq.u(z, z0=0), r"$u_0$"],
            [x / 10, -analyzer.sm.fq.w(z), r"$-w$"],
            [x / 10, np.rad2deg(analyzer.sm.fq.psi(z)), r"$\psi$"],
        ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            ax1label=r"Displacements (mm)",
            ax1data=data,
            name="fea_disp",
            labelpos=-50,
        )

    def plot_fea_stress(
        self, analyzer: Analyzer, xb: np.ndarray, zb: np.ndarray, fea: np.ndarray
    ):
        """Wrap stress plot."""
        data = [
            [fea[:, 0] / 10, 1e3 * np.flipud(fea[:, 2]), r"FEA $\sigma_2$"],
            [fea[:, 0] / 10, 1e3 * np.flipud(fea[:, 3]), r"FEA $\tau_{12}$"],
            [xb / 10, analyzer.sm.fq.tau(zb, unit="kPa"), r"$\tau$"],
            [xb / 10, analyzer.sm.fq.sig(zb, unit="kPa"), r"$\sigma$"],
        ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            ax1label=r"Stress (kPa)",
            ax1data=data,
            name="fea_stress",
            labelpos=-50,
        )

    # === BASE PLOT FUNCTION ======================================================

    def _plot_data(
        self,
        scenario: Scenario,
        name,
        ax1data,
        ax1label,
        ax2data=None,
        ax2label=None,
        labelpos=None,
        vlines=True,
        xlabel=r"Horizontal position $x$ (cm)",
    ):
        """Plot data. Base function."""
        # Figure setup
        plt.rcdefaults()
        plt.rc("font", family="serif", size=10)
        plt.rc("mathtext", fontset="cm")

        # Create figure
        fig = plt.figure(figsize=(4, 8 / 3))
        ax1 = fig.gca()

        # Axis limits
        ax1.autoscale(axis="x", tight=True)

        # Set axis labels
        ax1.set_xlabel(xlabel + r" $\longrightarrow$")
        ax1.set_ylabel(ax1label + r" $\longrightarrow$")

        # Plot x-axis
        ax1.axhline(0, linewidth=0.5, color="gray")

        ki = scenario.ki
        li = scenario.li
        mi = scenario.mi

        # Plot vertical separators
        if vlines:
            ax1.axvline(0, linewidth=0.5, color="gray")
            for i, f in enumerate(ki):
                if not f:
                    ax1.axvspan(
                        sum(li[:i]) / 10,
                        sum(li[: i + 1]) / 10,
                        facecolor="gray",
                        alpha=0.05,
                        zorder=100,
                    )
            for i, m in enumerate(mi, start=1):
                if m > 0:
                    ax1.axvline(sum(li[:i]) / 10, linewidth=0.5, color="gray")
        else:
            ax1.autoscale(axis="y", tight=True)

        # Calculate labelposition
        if not labelpos:
            x = ax1data[0][0]
            labelpos = int(0.95 * len(x[~np.isnan(x)]))

        # Fill left y-axis
        i = 0
        for x, y, label in ax1data:
            i += 1
            if label == "" or "FEA" in label:
                # line, = ax1.plot(x, y, 'k:', linewidth=1)
                ax1.plot(x, y, linewidth=3, color="white")
                (line,) = ax1.plot(x, y, ":", linewidth=1)  # , color='black'
                thislabelpos = -2
                x, y = x[~np.isnan(x)], y[~np.isnan(x)]
                xtx = (x[thislabelpos - 1] + x[thislabelpos]) / 2
                ytx = (y[thislabelpos - 1] + y[thislabelpos]) / 2
                ax1.text(xtx, ytx, label, color=line.get_color(), **LABELSTYLE)
            else:
                # Plot line
                ax1.plot(x, y, linewidth=3, color="white")
                (line,) = ax1.plot(x, y, linewidth=1)
                # Line label
                x, y = x[~np.isnan(x)], y[~np.isnan(x)]
                if len(x) > 0:
                    xtx = (x[labelpos - 10 * i - 1] + x[labelpos - 10 * i]) / 2
                    ytx = (y[labelpos - 10 * i - 1] + y[labelpos - 10 * i]) / 2
                    ax1.text(xtx, ytx, label, color=line.get_color(), **LABELSTYLE)

        # Fill right y-axis
        if ax2data:
            # Create right y-axis
            ax2 = ax1.twinx()
            # Set axis label
            ax2.set_ylabel(ax2label + r" $\longrightarrow$")
            # Fill
            for x, y, label in ax2data:
                # Plot line
                ax2.plot(x, y, linewidth=3, color="white")
                (line,) = ax2.plot(x, y, linewidth=1, color=COLORS[8, 0])
                # Line label
                x, y = x[~np.isnan(x)], y[~np.isnan(x)]
                xtx = (x[labelpos - 1] + x[labelpos]) / 2
                ytx = (y[labelpos - 1] + y[labelpos]) / 2
                ax2.text(xtx, ytx, label, color=line.get_color(), **LABELSTYLE)

        # Save figure
        self._save_figure(name, fig)

        # Reset plot styles
        plt.rcdefaults()
