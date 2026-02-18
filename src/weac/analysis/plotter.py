"""
This module provides plotting functions for visualizing the results of the WEAC model.
"""

# Standard library imports
import colorsys
import logging
import os
from typing import Literal

# Third party imports
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon, Rectangle
from scipy.optimize import brentq

from weac.analysis.analyzer import Analyzer
from weac.analysis.criteria_evaluator import (
    CoupledCriterionResult,
    CriteriaEvaluator,
    FindMinimumForceResult,
)

# Module imports
from weac.components.layer import WeakLayer
from weac.core.scenario import Scenario
from weac.core.slab import Slab
from weac.core.system_model import SystemModel
from weac.utils.misc import isnotebook

logger = logging.getLogger(__name__)

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


def _significant_digits(decimal: float) -> int:
    """Return the number of significant digits for a given decimal."""
    if decimal == 0:
        return 1
    try:
        sig_digits = -int(np.floor(np.log10(decimal)))
    except ValueError:
        sig_digits = 3
    return sig_digits


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
        self.labels = LABELSTYLE
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
        system_model: SystemModel | None = None,
        system_models: list[SystemModel] | None = None,
    ) -> list[SystemModel]:
        """Determine which systems to plot based on override parameters."""
        if system_model is not None and system_models is not None:
            raise ValueError(
                "Provide either 'system_model' or 'system_models', not both"
            )
        if isinstance(system_model, SystemModel):
            return [system_model]
        if isinstance(system_models, list):
            return system_models
        raise ValueError(
            "Must provide either 'system_model' or 'system_models' as a "
            "SystemModel or list of SystemModels"
        )

    def _save_figure(self, filename: str, fig: Figure | None = None):
        """Save figure with proper formatting."""
        if fig is None:
            fig = plt.gcf()

        filepath = os.path.join(self.plot_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")

        if not isnotebook():
            plt.close(fig)

    def plot_slab_profile(
        self,
        weak_layers: list[WeakLayer] | WeakLayer,
        slabs: list[Slab] | Slab,
        filename: str = "slab_profile",
        labels: list[str] | str | None = None,
        colors: list[str] | None = None,
    ):
        """
        Plot slab layer profiles for comparison.

        Parameters
        ----------
        weak_layers : list[WeakLayer] | WeakLayer
            The weak layer or layers to plot.
        slabs : list[Slab] | Slab
            The slab or slabs to plot.
        filename : str, optional
            Filename for saving plot
        labels : list of str, optional
            Labels for each system.
        colors : list of str, optional
            Colors for each system.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.
        """
        if isinstance(weak_layers, WeakLayer):
            weak_layers = [weak_layers]
        if isinstance(slabs, Slab):
            slabs = [slabs]

        if len(weak_layers) != len(slabs):
            raise ValueError("Number of weak layers must match number of slabs")

        if labels is None:
            labels = [f"System {i + 1}" for i in range(len(weak_layers))]
        elif isinstance(labels, str):
            labels = [labels] * len(slabs)
        elif len(labels) != len(slabs):
            raise ValueError("Number of labels must match number of slabs")

        if colors is None:
            plot_colors = [self.colors[i, 0] for i in range(len(slabs))]
        else:
            plot_colors = colors

        # Plot Setup
        plt.rcdefaults()
        plt.rc("font", family="serif", size=8)
        plt.rc("mathtext", fontset="cm")

        fig = plt.figure(figsize=(8 / 3, 4))
        ax1 = fig.gca()

        # Plot 1: Layer thickness and density
        max_height = 0
        for i, slab in enumerate(slabs):
            total_height = slab.H + weak_layers[i].h
            max_height = max(max_height, total_height)

        for i, (weak_layer, slab, label, color) in enumerate(
            zip(weak_layers, slabs, labels, plot_colors)
        ):
            # Plot weak layer
            wl_y = [-weak_layer.h, 0]
            wl_x = [weak_layer.rho, weak_layer.rho]
            ax1.fill_betweenx(wl_y, 0, wl_x, color="red", alpha=0.8, hatch="///")

            # Plot slab layers
            x_coords = []
            y_coords = []
            current_height = 0

            # As slab.layers is top-down
            for layer in reversed(slab.layers):
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

        handles, slab_labels = ax1.get_legend_handles_labels()
        weak_layer_patch = Patch(
            facecolor="red", alpha=0.8, hatch="///", label="Weak Layer"
        )
        ax1.legend(
            handles=[weak_layer_patch] + handles, labels=["Weak Layer"] + slab_labels
        )

        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(500, 0)
        ax1.set_ylim(-min(weak_layer.h for weak_layer in weak_layers), max_height)

        if filename:
            self._save_figure(filename, fig)

        return fig

    def plot_rotated_slab_profile(
        self,
        weak_layer: WeakLayer,
        slab: Slab,
        angle: float = 0,
        weight: float = 0,
        slab_width: float = 200,
        filename: str = "rotated_slab_profile",
        title: str = "Rotated Slab Profile",
    ):
        """
        Plot a rectangular slab profile with layers stacked vertically, colored by density,
        and rotated by the specified angle.

        Parameters
        ----------
        weak_layer : WeakLayer
            The weak layer to plot at the bottom.
        slab : Slab
            The slab with layers to plot.
        angle : float, optional
            Rotation angle in degrees. Default is 0.
        slab_width : float, optional
            Width of the slab rectangle in mm. Default is 200.
        filename : str, optional
            Filename for saving plot. Default is "rotated_slab_profile".
        title : str, optional
            Plot title. Default is "Rotated Slab Profile".

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.
        """
        # Plot Setup
        plt.rcdefaults()
        plt.rc("font", family="serif", size=10)
        plt.rc("mathtext", fontset="cm")

        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.gca()

        # Calculate total height
        total_height = slab.H + weak_layer.h

        # Create density-based colormap
        all_densities = [weak_layer.rho] + [layer.rho for layer in slab.layers]
        min_density = min(all_densities)
        max_density = max(all_densities)

        # Normalize densities for color mapping
        norm = mc.Normalize(vmin=min_density, vmax=max_density)
        cmap = plt.get_cmap("viridis")  # You can change this to any colormap

        # Function to create sloped layer (parallelogram)
        def create_sloped_layer(x, y, width, height, angle_rad):
            """Create a layer that follows the slope angle"""
            # Calculate horizontal offset for the slope
            slope_offset = width * np.sin(angle_rad)

            # Create parallelogram corners
            # Bottom edge is horizontal, top edge is shifted by slope_offset
            corners = np.array(
                [
                    [x, y],  # Bottom left
                    [x + width, y + slope_offset],  # Bottom right
                    [x + width, y + height + slope_offset],  # Top right (shifted)
                    [x, y + height],  # Top left (shifted)
                ]
            )

            return corners

        # Convert angle to radians
        angle_rad = np.radians(angle)

        # Start from bottom (weak layer)
        current_y = 0

        # Plot weak layer
        wl_corners = create_sloped_layer(
            0, current_y, slab_width, weak_layer.h, angle_rad
        )
        wl_color = cmap(norm(weak_layer.rho))
        wl_patch = Polygon(
            wl_corners,
            facecolor=wl_color,
            edgecolor="black",
            linewidth=1,
            alpha=0.8,
            hatch="///",
        )
        ax.add_patch(wl_patch)

        # Add density label for weak layer
        wl_center = np.mean(wl_corners, axis=0)
        ax.text(
            wl_center[0],
            wl_center[1],
            f"{weak_layer.rho:.0f}\nkg/m³",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

        current_y += weak_layer.h

        # Plot slab layers (from bottom to top)
        top_layer_corners = None
        for _i, layer in enumerate(reversed(slab.layers)):
            layer_corners = create_sloped_layer(
                0, current_y, slab_width, layer.h, angle_rad
            )
            layer_color = cmap(norm(layer.rho))
            layer_patch = Polygon(
                layer_corners,
                facecolor=layer_color,
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
            )
            ax.add_patch(layer_patch)

            # Add density label for slab layer
            layer_center = np.mean(layer_corners, axis=0)
            ax.text(
                layer_center[0],
                layer_center[1],
                f"{layer.rho:.0f}\nkg/m³",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

            current_y += layer.h
            # Keep track of the top layer corners for arrow placement
            top_layer_corners = layer_corners

        # Add weight arrow if weight > 0 and we have layers
        if weight > 0 and top_layer_corners is not None:
            # Calculate midpoint of top edge of highest layer
            # Top edge is between points 2 and 3 (top right and top left)
            top_left = top_layer_corners[3]
            top_right = top_layer_corners[2]
            arrow_start_x = (top_left[0] + top_right[0]) / 2
            arrow_start_y = (top_left[1] + top_right[1]) / 2

            # Scale arrow based on weight (0-400 maps to 0-100, above 400 = 100)
            max_arrow_height = 100
            arrow_height = min(weight * max_arrow_height / 400, max_arrow_height)
            arrow_width = arrow_height * 0.3  # Arrow width proportional to height

            # Create arrow pointing downward
            arrow_tip_x = arrow_start_x
            arrow_tip_y = arrow_start_y

            # Arrow shaft (rectangular part)
            shaft_width = arrow_width * 0.3
            shaft_left = arrow_start_x - shaft_width / 2
            shaft_right = arrow_start_x + shaft_width / 2
            shaft_top = arrow_start_y + arrow_height
            shaft_bottom = arrow_tip_y + arrow_width * 0.4

            # Arrow head (triangular part)
            head_left = arrow_start_x - arrow_width / 2
            head_right = arrow_start_x + arrow_width / 2
            head_top = shaft_bottom

            # Draw arrow shaft
            shaft_corners = np.array(
                [
                    [shaft_left, shaft_top],
                    [shaft_right, shaft_top],
                    [shaft_right, shaft_bottom],
                    [shaft_left, shaft_bottom],
                ]
            )
            shaft_patch = Polygon(
                shaft_corners,
                facecolor="red",
                edgecolor="darkred",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(shaft_patch)

            # Draw arrow head
            head_corners = np.array(
                [
                    [head_left, head_top],
                    [head_right, head_top],
                    [arrow_tip_x, arrow_tip_y],
                ]
            )
            head_patch = Polygon(
                head_corners,
                facecolor="red",
                edgecolor="darkred",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(head_patch)

            # Add weight label
            ax.text(
                arrow_start_x + arrow_width * 0.7,
                arrow_start_y - arrow_height / 2,
                f"{weight:.0f} kg",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="darkred",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.8,
                },
            )

        # Calculate plot limits to accommodate rotated rectangle
        margin = max(slab_width, total_height) * 0.2

        # Find the bounds of all rotated rectangles
        all_corners = []
        current_y = 0

        # Weak layer corners
        wl_corners = create_sloped_layer(
            0, current_y, slab_width, weak_layer.h, angle_rad
        )
        all_corners.extend(wl_corners)
        current_y += weak_layer.h

        # Slab layer corners
        for layer in reversed(slab.layers):
            layer_corners = create_sloped_layer(
                0, current_y, slab_width, layer.h, angle_rad
            )
            all_corners.extend(layer_corners)
            current_y += layer.h

        all_corners = np.array(all_corners)
        min_x, max_x = all_corners[:, 0].min(), all_corners[:, 0].max()
        min_y, max_y = all_corners[:, 1].min(), all_corners[:, 1].max()

        # Set axis limits with margin
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)

        # Set labels and title
        ax.set_xlabel("Width (mm)")
        ax.set_ylabel("Height (mm)")
        ax.set_title(f"{title}\nSlope Angle: {angle}°")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Density (kg/m³)")

        # Add legend
        weak_layer_patch = Patch(
            facecolor=cmap(norm(weak_layer.rho)),
            hatch="///",
            edgecolor="black",
            label="Weak Layer",
        )
        slab_patch = Patch(facecolor="gray", edgecolor="black", label="Slab Layers")
        ax.legend(handles=[weak_layer_patch, slab_patch], loc="upper right")

        # Equal aspect ratio and grid
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Remove axis ticks for cleaner look
        ax.tick_params(axis="both", which="major", labelsize=8)

        plt.tight_layout()

        if filename:
            self._save_figure(filename, fig)

        return fig

    def plot_section_forces(
        self,
        system_model: SystemModel | None = None,
        system_models: list[SystemModel] | None = None,
        filename: str = "section_forces",
        labels: list[str] | None = None,
        colors: list[str] | None = None,
    ):
        """
        Plot section forces (N, M, V) for comparison.

        Parameters
        ----------
        system_model : SystemModel, optional
            Single system to plot (overrides default)
        system_models : list[SystemModel], optional
            Multiple systems to plot (overrides default)
        filename : str, optional
            Filename for saving plot
        labels : list of str, optional
            Labels for each system.
        colors : list of str, optional
            Colors for each system.
        """
        systems_to_plot = self._get_systems_to_plot(system_model, system_models)

        if labels is None:
            labels = [f"System {i + 1}" for i in range(len(systems_to_plot))]
        if colors is None:
            plot_colors = [self.colors[i, 0] for i in range(len(systems_to_plot))]
        else:
            plot_colors = colors

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        for i, system in enumerate(systems_to_plot):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            fq = system.fq

            # Convert x to meters for plotting
            x_m = x / 1000

            # Plot axial force N
            N = fq.N(z)
            axes[0].plot(x_m, N, color=plot_colors[i], label=labels[i], linewidth=2)

            # Plot bending moment M
            M = fq.M(z)
            axes[1].plot(x_m, M, color=plot_colors[i], label=labels[i], linewidth=2)

            # Plot shear force V
            V = fq.V(z)
            axes[2].plot(x_m, V, color=plot_colors[i], label=labels[i], linewidth=2)

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
        system_model: SystemModel | None = None,
        system_models: list[SystemModel] | None = None,
        filename: str = "ERR",
        labels: list[str] | None = None,
        colors: list[str] | None = None,
    ):
        """
        Plot energy release rates (G_I, G_II) for comparison.

        Parameters
        ----------
        system_model : SystemModel, optional
            Single system to plot (overrides default)
        system_models : list[SystemModel], optional
            Multiple systems to plot (overrides default)
        filename : str, optional
            Filename for saving plot
        labels : list of str, optional
            Labels for each system.
        colors : list of str, optional
            Colors for each system.
        """
        systems_to_plot = self._get_systems_to_plot(system_model, system_models)

        if labels is None:
            labels = [f"System {i + 1}" for i in range(len(systems_to_plot))]
        if colors is None:
            plot_colors = [self.colors[i, 0] for i in range(len(systems_to_plot))]
        else:
            plot_colors = colors

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        for i, system in enumerate(systems_to_plot):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            fq = system.fq

            # Convert x to meters for plotting
            x_m = x / 1000

            # Plot Mode I energy release rate
            G_I = fq.Gi(z, unit="kJ/m^2")
            axes[0].plot(x_m, G_I, color=plot_colors[i], label=labels[i], linewidth=2)

            # Plot Mode II energy release rate
            G_II = fq.Gii(z, unit="kJ/m^2")
            axes[1].plot(x_m, G_II, color=plot_colors[i], label=labels[i], linewidth=2)

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
        window: float = np.inf,
        pad: int = 2,
        levels: int = 300,
        aspect: int = 2,
        field: Literal["w", "u", "principal", "Sxx", "Txz", "Szz"] = "w",
        normalize: bool = True,
        filename: str = "deformed_slab",
    ) -> Figure:
        """
        Plot deformed slab with field contours.

        Parameters
        ----------
        xsl : np.ndarray
            Slab x-coordinates.
        xwl : np.ndarray
            Weak layer x-coordinates.
        z : np.ndarray
            Solution vector.
        analyzer : Analyzer
            Analyzer instance.
        dz : int, optional
            Element size along z-axis (mm). Default is 2 mm.
        scale : int, optional
            Deformation scale factor. Default is 100.
        window : float, optional
            Plot window width. Default is inf.
        pad : int, optional
            Padding around plot. Default is 2.
        levels : int, optional
            Number of contour levels. Default is 300.
        aspect : int, optional
            Aspect ratio. Default is 2.
        field : str, optional
            Field to plot ('w', 'u', 'principal', 'Sxx', 'Txz', 'Szz'). Default is 'w'.
        normalize : bool, optional
            Toggle normalization. Default is True.
        filename : str, optional
            Filename for saving plot. Default is "deformed_slab".

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        zi = analyzer.get_zmesh(dz=dz)["z"]
        H = analyzer.sm.slab.H
        phi = analyzer.sm.scenario.phi
        system_type = analyzer.sm.scenario.system_type
        fq = analyzer.sm.fq

        # Compute slab displacements on grid (cm)
        Usl = np.vstack([fq.u(z, h0=h0, unit="cm") for h0 in zi])
        Wsl = np.vstack([fq.w(z, unit="cm") for _ in zi])
        Sigmawl = np.where(np.isfinite(xwl), fq.sig(z, unit="kPa"), np.nan)
        Tauwl = np.where(np.isfinite(xwl), fq.tau(z, unit="kPa"), np.nan)

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
            zmax = np.max(Zsl + scale * Wsl) + pad
        zmin = np.min(Zsl) - pad

        # Filter out NaN values from weak layer coordinates
        nanmask = np.isfinite(xwl)
        xwl_finite = xwl[nanmask]

        # Compute weak-layer grid coordinates (cm) - only for finite xwl
        Xwl, Zwl = np.meshgrid(1e-1 * xwl_finite, [1e-1 * (zi[-1] + H / 2), zmax])

        # Assemble weak-layer displacement field (top and bottom) - only for finite xwl
        Uwl = np.vstack([Usl[-1, nanmask], np.zeros(xwl_finite.shape[0])])
        Wwl = np.vstack([Wsl[-1, nanmask], np.zeros(xwl_finite.shape[0])])

        # Compute stress or displacement fields
        match field:
            # Horizontal displacements (um)
            case "u":
                slab = 1e4 * Usl
                weak = 1e4 * Usl[-1, nanmask]
                label = r"$u$ ($\mu$m)"
            # Vertical deflection (um)
            case "w":
                slab = 1e4 * Wsl
                weak = 1e4 * Wsl[-1, nanmask]
                label = r"$w$ ($\mu$m)"
            case "Sxx":
                slab = analyzer.Sxx(z, phi, dz=dz, unit="kPa", normalize=normalize)
                weak = np.zeros(xwl_finite.shape[0])
                label = (
                    r"$\sigma_{xx}/\sigma_+$" if normalize else r"$\sigma_{xx}$ (kPa)"
                )
            # Shear stresses (kPa)
            case "Txz":
                slab = analyzer.Txz(z, phi, dz=dz, unit="kPa", normalize=normalize)
                weak = Tauwl[nanmask]
                label = r"$\tau_{xz}/\sigma_+$" if normalize else r"$\tau_{xz}$ (kPa)"
            # Transverse normal stresses (kPa)
            case "Szz":
                slab = analyzer.Szz(z, phi, dz=dz, unit="kPa", normalize=normalize)
                weak = Sigmawl[nanmask]
                label = (
                    r"$\sigma_{zz}/\sigma_+$" if normalize else r"$\sigma_{zz}$ (kPa)"
                )
            # Principal stresses
            case "principal":
                slab = analyzer.principal_stress_slab(
                    z, phi, dz=dz, val="max", unit="kPa", normalize=normalize
                )
                weak_full = analyzer.principal_stress_weaklayer(
                    z, val="min", unit="kPa", normalize=normalize
                )
                weak = weak_full[nanmask]
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
        ax.axhline(zmax, color="k", linewidth=1)

        # Plot outlines of the undeformed and deformed slab
        ax.plot(_outline(Xsl), _outline(Zsl), "k--", alpha=0.3, linewidth=1)
        ax.plot(
            _outline(Xsl + scale * Usl), _outline(Zsl + scale * Wsl), "k", linewidth=1
        )

        # Plot deformed weak-layer _outline
        if system_type in ["-pst", "pst-", "-vpst", "vpst-"]:
            ax.plot(
                _outline(Xwl + scale * Uwl),
                _outline(Zwl + scale * Wwl),
                "k",
                linewidth=1,
            )

        cmap = plt.get_cmap("RdBu_r")
        cmap.set_over(_adjust_lightness(cmap(1.0), 0.9))
        cmap.set_under(_adjust_lightness(cmap(0.0), 0.9))

        # Plot fields
        ax.contourf(
            Xsl + scale * Usl,
            Zsl + scale * Wsl,
            slab,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        ax.contourf(
            Xwl + scale * Uwl,
            Zwl + scale * Wwl,
            weak,
            levels=levels,
            cmap=cmap,
            extend="both",
        )

        # Plot setup
        ax.axis("scaled")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([zmin, zmax])
        ax.set_aspect(aspect)
        ax.invert_yaxis()
        ax.use_sticky_edges = False

        # Plot labels
        ax.set_xlabel(r"lateral position $x$ (cm) $\longrightarrow$")
        ax.set_ylabel("depth below surface\n" + r"$\longleftarrow $ $d$ (cm)")
        ax.set_title(rf"${scale}\!\times\!$ scaled deformations (cm)", size=10)

        # Show colorbar
        ticks = np.linspace(levels[0], levels[-1], num=11, endpoint=True)
        fig.colorbar(
            ax.contourf(
                Xsl + scale * Usl,
                Zsl + scale * Wsl,
                slab,
                levels=levels,
                cmap=cmap,
                extend="both",
            ),
            orientation="horizontal",
            ticks=ticks,
            label=label,
            aspect=35,
        )

        # Save figure
        self._save_figure(filename, fig)

        return fig

    def plot_visualize_deformation(
        self,
        xsl: np.ndarray,
        xwl: np.ndarray,
        z: np.ndarray,
        analyzer: Analyzer,
        window: float | None = None,
        weaklayer_proportion: float | None = None,
        dz: int = 2,
        levels: int = 300,
        field: Literal["w", "u", "principal", "Sxx", "Txz", "Szz"] = "w",
        normalize: bool = True,
        filename: str = "visualize_deformation",
    ) -> Figure:
        """
        Plot visualize deformation of the slab and weak layer.

        Parameters
        ----------
        xsl : np.ndarray
            Slab x-coordinates.
        xwl : np.ndarray
            Weak layer x-coordinates.
        z : np.ndarray
            Solution vector.
        analyzer : Analyzer
            Analyzer instance.
        window: float | None, optional
            Window size for the plot. Shows the right edge of the slab, where the slab is deformed. Default is None.
        weaklayer_proportion: float | None, optional
            Proportion of the plot to allocate to the weak layer. Default is None.
        dz : int, optional
            Element size along z-axis (mm). Default is 2 mm.
        levels : int, optional
            Number of levels for the colormap. Default is 300.
        field : str, optional
            Field to plot ('w', 'u', 'principal', 'Sxx', 'Txz', 'Szz'). Default is 'w'.
        normalize : bool, optional
            Toggle normalization. Default is True.
        filename : str, optional
            Filename for saving plot. Default is "visualize_deformation".

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        zi = analyzer.get_zmesh(dz=dz)["z"]
        H = analyzer.sm.slab.H
        phi = analyzer.sm.scenario.phi
        system_type = analyzer.sm.scenario.system_type
        fq = analyzer.sm.fq
        sigma_comp = (
            analyzer.sm.weak_layer.sigma_comp
        )  # Compressive strength of the weak layer [kPa]

        # Compute slab displacements on grid (cm)
        Usl = np.vstack([fq.u(z, h0=h0, unit="cm") for h0 in zi])
        Wsl = np.vstack([fq.w(z, unit="cm") for _ in zi])
        Sigmawl = np.where(np.isfinite(xwl), fq.sig(z, unit="kPa"), np.nan)
        Tauwl = np.where(np.isfinite(xwl), fq.tau(z, unit="kPa"), np.nan)

        # Put coordinate origin at horizontal center
        if system_type in ["skier", "skiers"]:
            xsl = xsl - max(xsl) / 2
            xwl = xwl - max(xwl) / 2

        # Physical dimensions in cm
        H_cm = H * 1e-1  # Slab height in cm
        h_cm = analyzer.sm.weak_layer.h * 1e-1  # Weak layer height in cm
        crack_h_cm = analyzer.sm.scenario.crack_h * 1e-1  # Crack height in cm

        # Compute slab grid coordinates with vertical origin at top surface (cm)
        Xsl, Zsl = np.meshgrid(1e-1 * (xsl), 1e-1 * (zi + H / 2))

        # Calculate maximum displacement first (needed for proportion calculation)
        max_w_displacement = np.nanmax(np.abs(Wsl))

        # Calculate dynamic proportions based on displacement
        # Weak layer percentage = weak_layer_height / max_displacement (as ratio)
        # But capped at 40% maximum
        if weaklayer_proportion is None:
            if max_w_displacement > 0:
                weaklayer_proportion = min(0.3, (h_cm / max_w_displacement) * 0.1)
            else:
                weaklayer_proportion = 0.3

        # Slab takes the remaining space
        slab_proportion = 1.0 - weaklayer_proportion
        cracked_ratio = crack_h_cm / h_cm
        cracked_proportion = weaklayer_proportion * cracked_ratio

        # Set up plot coordinate system
        # Plot height is normalized: slab (0 to slab_proportion), weak layer (slab_proportion to slab_proportion+weaklayer_proportion)
        total_height_plot = (
            slab_proportion + weaklayer_proportion
        )  # Total height without displacement
        # Map physical dimensions to plot coordinates
        deformation_scale = weaklayer_proportion / h_cm

        # Get x-axis limits spanning all provided x values (deformed and undeformed)
        xmax = np.max([np.max(Xsl), np.max(Xsl + deformation_scale * Usl)]) + 10.0
        xmin = np.min([np.min(Xsl), np.min(Xsl + deformation_scale * Usl)]) - 10.0

        # Calculate zmax including maximum deformation
        zmax = total_height_plot

        # Convert physical coordinates to plot coordinates for slab
        # Zsl is in cm, we need to map it to plot coordinates (0 to slab_proportion)
        Zsl_plot = (Zsl / H_cm) * slab_proportion

        # Filter out NaN values from weak layer coordinates
        nanmask = np.isfinite(xwl)
        xwl_finite = xwl[nanmask]

        # Compute weak-layer grid coordinates in plot units
        # Weak layer extends from bottom of slab (slab_proportion) to total height (1.0)
        Xwl, Zwl_plot = np.meshgrid(
            1e-1 * xwl_finite, [slab_proportion, total_height_plot]
        )

        # Assemble weak-layer displacement field (top and bottom) - only for finite xwl
        Uwl = np.vstack([Usl[-1, nanmask], np.zeros(xwl_finite.shape[0])])
        Wwl = np.vstack([Wsl[-1, nanmask], np.zeros(xwl_finite.shape[0])])

        # Convert slab displacements to plot coordinates
        # Scale factor for displacements:
        # So scaled displacement in plot units = scale * Wsl
        Wsl_plot = (
            deformation_scale * Wsl
        )  # Already in plot units (proportion of total height)
        Usl_plot = deformation_scale * Usl  # Horizontal displacements also scaled
        Wwl_plot = deformation_scale * Wwl  # Weak layer displacements
        Uwl_plot = deformation_scale * Uwl  # Weak layer horizontal displacements

        # Compute stress or displacement fields
        match field:
            # Horizontal displacements (um)
            case "u":
                slab = 1e4 * Usl
                weak = 1e4 * Usl[-1, nanmask]
                label = r"$u$ ($\mu$m)"
            # Vertical deflection (um)
            case "w":
                slab = 1e4 * Wsl
                weak = 1e4 * Wsl[-1, nanmask]
                label = r"$w$ ($\mu$m)"
            # Axial normal stresses (kPa)
            case "Sxx":
                slab = analyzer.Sxx(z, phi, dz=dz, unit="kPa", normalize=normalize)
                weak = np.zeros(xwl_finite.shape[0])
                label = (
                    r"$\sigma_{xx}/\sigma_+$" if normalize else r"$\sigma_{xx}$ (kPa)"
                )
            # Shear stresses (kPa)
            case "Txz":
                slab = analyzer.Txz(z, phi, dz=dz, unit="kPa", normalize=normalize)
                weak = Tauwl[nanmask]
                label = r"$\tau_{xz}/\sigma_+$" if normalize else r"$\tau_{xz}$ (kPa)"
            # Transverse normal stresses (kPa)
            case "Szz":
                slab = analyzer.Szz(z, phi, dz=dz, unit="kPa", normalize=normalize)
                weak = Sigmawl[nanmask]
                label = (
                    r"$\sigma_{zz}/\sigma_+$" if normalize else r"$\sigma_{zz}$ (kPa)"
                )
            # Principal stresses
            case "principal":
                slab = analyzer.principal_stress_slab(
                    z, phi, dz=dz, val="max", unit="kPa", normalize=normalize
                )
                weak_full = analyzer.principal_stress_weaklayer(
                    z, sc=sigma_comp, val="min", unit="kPa", normalize=normalize
                )
                weak = weak_full[nanmask]
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

        # Complement label
        label += r"  $\longrightarrow$"

        # Assemble weak-layer output on grid
        weak = np.vstack([weak, weak])

        # Normalize colormap
        absmax = np.nanmax(np.abs([slab.min(), slab.max(), weak.min(), weak.max()]))
        clim = np.round(absmax, _significant_digits(absmax))
        levels = np.linspace(-clim, clim, num=levels + 1, endpoint=True)

        # Plot baseline
        ax.axhline(zmax, color="k", linewidth=1)

        # Plot outlines of the undeformed and deformed slab (using plot coordinates)
        ax.plot(_outline(Xsl), _outline(Zsl_plot), "--", alpha=0.3, linewidth=1)
        ax.plot(
            _outline(Xsl + Usl_plot),
            _outline(Zsl_plot + Wsl_plot),
            "-",
            linewidth=1,
            color="k",
        )

        # Plot cracked weak-layer outline (where there is no weak layer)
        xwl_cracked = xsl[~nanmask]
        Xwl_cracked, Zwl_cracked_plot = np.meshgrid(
            1e-1 * xwl_cracked,
            [slab_proportion + cracked_proportion, total_height_plot],
        )
        # No displacements for the cracked weak layer outline (undeformed)
        if xwl_cracked.shape[0] > 0:
            ax.plot(
                _outline(Xwl_cracked),
                _outline(Zwl_cracked_plot),
                "k-",
                alpha=0.3,
                linewidth=1,
            )

        # Then plot the deformed weak-layer outline where it exists
        if system_type in ["-pst", "pst-", "-vpst", "vpst-"]:
            ax.plot(
                _outline(Xwl + Uwl_plot),
                _outline(Zwl_plot + Wwl_plot),
                "k",
                linewidth=1,
            )

        cmap = plt.get_cmap("RdBu_r")
        cmap.set_over(_adjust_lightness(cmap(1.0), 0.9))
        cmap.set_under(_adjust_lightness(cmap(0.0), 0.9))

        # Plot fields (using plot coordinates)
        ax.contourf(
            Xsl + Usl_plot,
            Zsl_plot + Wsl_plot,
            slab,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        ax.contourf(
            Xwl + Uwl_plot,
            Zwl_plot + Wwl_plot,
            weak,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        if xwl_cracked.shape[0] > 0:
            ax.contourf(
                Xwl_cracked,
                Zwl_cracked_plot,
                np.zeros((2, xwl_cracked.shape[0])),
                levels=levels,
                cmap=cmap,
                extend="both",
            )

        # Plot setup
        # Set y-limits to match plot coordinate system (0 to total_height_plot = 1.0)
        plot_ymin = -0.1
        plot_ymax = (
            total_height_plot  # Should be 1.0 (slab_proportion + weaklayer_proportion)
        )

        # Set limits first, then aspect ratio to avoid matplotlib adjusting limits
        if window is None:
            ax.set_xlim([xmin, xmax])
        else:
            ax.set_xlim([xmax - window, xmax])
        ax.set_ylim([plot_ymin, plot_ymax])
        ax.invert_yaxis()
        ax.use_sticky_edges = False

        # Hide the default y-axis on main axis (we'll use custom axes)
        ax.yaxis.set_visible(False)

        # Set up dual y-axes
        # Right axis: slab height in cm (0 at top, H_cm at bottom of slab)
        ax_right = ax.twinx()
        slab_height_max = H_cm
        # Map plot coordinates to physical slab height values
        # Plot: 0 to slab_proportion (0.6) maps to physical: 0 to H_cm
        slab_height_ticks = np.linspace(0, slab_height_max, num=5)
        slab_height_positions_plot = (
            slab_height_ticks / slab_height_max
        ) * slab_proportion
        ax_right.set_yticks(slab_height_positions_plot)
        ax_right.set_yticklabels([f"{tick:.1f}" for tick in slab_height_ticks])
        # Ensure right axis ticks and label are on the right side
        ax_right.yaxis.tick_right()
        ax_right.yaxis.set_label_position("right")
        ax_right.set_ylim([plot_ymin, plot_ymax])
        ax_right.invert_yaxis()
        ax_right.set_ylabel(
            r"slab depth [cm] $\longleftarrow$", rotation=90, labelpad=5, loc="top"
        )

        # Left axis: weak layer height in mm (0 at bottom of slab, h at bottom of weak layer)
        ax_left = ax.twinx()
        weak_layer_h_mm = analyzer.sm.weak_layer.h
        # Map plot coordinates to physical weak layer height values
        # Plot: slab_proportion (0.6) to total_height_plot (1.0) maps to physical: 0 to h_mm
        weaklayer_height_ticks = np.linspace(0, weak_layer_h_mm, num=3)
        # Map from plot coordinates (slab_proportion to 1.0) to physical (0 to h_mm)
        weaklayer_height_positions_plot = (
            slab_proportion
            + (weaklayer_height_ticks / weak_layer_h_mm) * weaklayer_proportion
        )
        ax_left.set_yticks(weaklayer_height_positions_plot)
        ax_left.set_yticklabels([f"{tick:.1f}" for tick in weaklayer_height_ticks])
        # Move left axis to the left side
        ax_left.yaxis.tick_left()
        ax_left.yaxis.set_label_position("left")
        ax_left.set_ylim([plot_ymin, plot_ymax])
        ax_left.invert_yaxis()
        ax_left.set_ylabel(
            r"weaklayer depth [mm] $\longleftarrow$",
            rotation=90,
            labelpad=5,
            loc="bottom",
        )

        # Plot labels
        ax.set_xlabel(r"lateral position $x$ (cm) $\longrightarrow$")
        ax.set_title(
            f"{field}{' (normalized to tensile strength)' if normalize else ''}",
            size=10,
        )

        # Show colorbar
        ticks = np.linspace(levels[0], levels[-1], num=11, endpoint=True)
        fig.colorbar(
            ax.contourf(
                Xsl + Usl_plot,
                Zsl_plot + Wsl_plot,
                slab,
                levels=levels,
                cmap=cmap,
                extend="both",
            ),
            orientation="horizontal",
            ticks=ticks,
            label=label,
            aspect=35,
        )

        # Save figure
        self._save_figure(filename, fig)

        return fig

    def plot_stress_envelope(
        self,
        system_model: SystemModel,
        criteria_evaluator: CriteriaEvaluator,
        all_envelopes: bool = False,
        filename: str | None = None,
    ):
        """
        Plot stress envelope in τ-σ space.

        Parameters
        ----------
        system_model : SystemModel
            System to plot
        criteria_evaluator : CriteriaEvaluator
            Criteria evaluator to use for the stress envelope
        all_envelopes : bool, optional
            Whether to plot all four quadrants of the envelope
        filename : str, optional
            Filename for saving plot
        """
        analyzer = self._get_analyzer(system_model)
        _, z, _ = analyzer.rasterize_solution(num=10000)
        fq = system_model.fq

        # Calculate stresses
        sigma = np.abs(fq.sig(z, unit="kPa"))
        tau = fq.tau(z, unit="kPa")

        fig, ax = plt.subplots(figsize=(4, 8 / 3))

        # Plot stress path
        ax.plot(sigma, tau, "b-", linewidth=2, label="Stress Path")
        ax.scatter(
            sigma[0], tau[0], color="green", s=10, marker="o", label="Start", zorder=5
        )
        ax.scatter(
            sigma[-1], tau[-1], color="red", s=10, marker="s", label="End", zorder=5
        )

        # --- Programmatic Envelope Calculation ---
        weak_layer = system_model.weak_layer

        # Define a function to find the root for a given tau
        def find_sigma_for_tau(tau_val, sigma_c, method: str | None = None):
            # Target function to find the root of: envelope(sigma, tau) - 1 = 0
            def envelope_root_func(sigma_val):
                return (
                    criteria_evaluator.stress_envelope(
                        sigma_val, tau_val, weak_layer, method=method
                    )
                    - 1
                )

            try:
                search_upper_bound = sigma_c * 1.1
                sigma_root = brentq(
                    envelope_root_func,
                    a=0,
                    b=search_upper_bound,
                    xtol=1e-6,
                    rtol=1e-6,
                )
                return sigma_root
            except ValueError:
                return np.nan

        # Calculate the corresponding sigma for each tau
        if all_envelopes:
            methods = [
                "mede_s-RG1",
                "mede_s-RG2",
                "mede_s-FCDH",
                "schottner",
                "adam_unpublished",
            ]
        else:
            methods = [criteria_evaluator.criteria_config.stress_envelope_method]

        colors = self.colors
        colors = np.array(colors)
        colors = np.tile(colors, (len(methods), 1))

        max_sigma = 0
        max_tau = 0
        for i, method in enumerate(methods):
            # Calculate tau_c for the given method to define tau_range
            config = criteria_evaluator.criteria_config
            density = weak_layer.rho
            tau_c = 0.0  # fallback
            sigma_c = 0.0
            if method == "adam_unpublished":
                scaling_factor = config.scaling_factor
                order_of_magnitude = config.order_of_magnitude
                if scaling_factor > 1:
                    order_of_magnitude = 0.7
                scaling_factor = max(scaling_factor, 0.55)

                tau_c = 5.09 * (scaling_factor**order_of_magnitude)
                sigma_c = 6.16 * (scaling_factor**order_of_magnitude)
            elif method == "schottner":
                rho_ice = 916.7
                sigma_y = 2000
                sigma_c_adam = 6.16
                tau_c_adam = 5.09
                order_of_magnitude = config.order_of_magnitude
                sigma_c = sigma_y * 13 * (density / rho_ice) ** order_of_magnitude
                tau_c = tau_c_adam * (sigma_c / sigma_c_adam)
                sigma_c = sigma_y * 13 * (density / rho_ice) ** order_of_magnitude
            elif method == "mede_s-RG1":
                tau_c = 3.53  # This is tau_T from Mede's paper
                sigma_c = 7.00
            elif method == "mede_s-RG2":
                tau_c = 1.22  # This is tau_T from Mede's paper
                sigma_c = 2.33
            elif method == "mede_s-FCDH":
                tau_c = 0.61  # This is tau_T from Mede's paper
                sigma_c = 1.49

            tau_range = np.linspace(0, tau_c, 100)
            sigma_envelope = np.array(
                [find_sigma_for_tau(t, sigma_c, method) for t in tau_range]
            )

            # Remove nan values where no root was found
            valid_points = ~np.isnan(sigma_envelope)
            valid_tau_range = tau_range[valid_points]
            sigma_envelope = sigma_envelope[valid_points]

            max_sigma = max(max_sigma, np.max(sigma_envelope))
            max_tau = max(max_tau, np.max(np.abs(valid_tau_range)))
            ax.plot(
                sigma_envelope,
                valid_tau_range,
                "--",
                linewidth=2,
                label=method,
                color=colors[i, 0],
            )
            ax.plot(
                -sigma_envelope, valid_tau_range, "--", linewidth=2, color=colors[i, 0]
            )
            ax.plot(
                -sigma_envelope,
                -valid_tau_range,
                "--",
                linewidth=2,
                color=colors[i, 0],
            )
            ax.plot(
                sigma_envelope, -valid_tau_range, "--", linewidth=2, color=colors[i, 0]
            )
            ax.scatter(0, tau_c, color="black", s=10, marker="o")
            ax.text(0, tau_c, r"$\tau_c$", color="black", ha="center", va="bottom")
            ax.scatter(sigma_c, 0, color="black", s=10, marker="o")
            ax.text(sigma_c, 0, r"$\sigma_c$", color="black", ha="left", va="center")

        # Formatting
        ax.set_xlabel("Compressive Strength σ (kPa)")
        ax.set_ylabel("Shear Strength τ (kPa)")
        ax.set_title("Weak Layer Stress Envelope")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

        max_tau = max(max_tau, float(np.max(np.abs(tau))))
        max_sigma = max(max_sigma, float(np.max(np.abs(sigma))))
        ax.set_xlim(0, max_sigma * 1.1)
        ax.set_ylim(-max_tau * 1.1, max_tau * 1.1)

        plt.tight_layout()

        if filename:
            self._save_figure(filename, fig)

        return fig

    def plot_err_envelope(
        self,
        system_model: SystemModel,
        criteria_evaluator: CriteriaEvaluator,
        filename: str = "err_envelope",
    ) -> Figure:
        """Plot the ERR envelope."""
        analyzer = self._get_analyzer(system_model)

        incr_energy = analyzer.incremental_ERR(unit="J/m^2")
        G_I = incr_energy[1]
        G_II = incr_energy[2]

        fig, ax = plt.subplots(figsize=(4, 8 / 3))

        # Plot stress path
        ax.scatter(
            np.abs(G_I),
            np.abs(G_II),
            color="blue",
            s=50,
            marker="o",
            label="Incremental ERR",
            zorder=5,
        )

        G_Ic = system_model.weak_layer.G_Ic
        G_IIc = system_model.weak_layer.G_IIc
        ax.scatter(0, G_IIc, color="black", s=100, marker="o", zorder=5)
        ax.text(
            0.01,
            G_IIc + 0.02,
            r"$G_{IIc}$",
            color="black",
            ha="left",
            va="center",
        )
        ax.scatter(G_Ic, 0, color="black", s=100, marker="o", zorder=5)
        ax.text(
            G_Ic + 0.01,
            0.01,
            r"$G_{Ic}$",
            color="black",
        )

        # --- Programmatic Envelope Calculation ---
        weak_layer = system_model.weak_layer

        # Define a function to find the root for a given G_II
        def find_GI_for_GII(GII_val):
            # Target function to find the root of: envelope(sigma, tau) - 1 = 0
            def envelope_root_func(GI_val):
                return (
                    criteria_evaluator.fracture_toughness_envelope(
                        GI_val,
                        GII_val,
                        weak_layer,
                    )
                    - 1
                )

            try:
                GI_root = brentq(envelope_root_func, a=0, b=50, xtol=1e-6, rtol=1e-6)
                return GI_root
            except ValueError:
                return np.nan

        # Generate a range of G values in the positive quadrant
        GII_max = system_model.weak_layer.G_IIc * 1.1
        GII_range = np.linspace(0, GII_max, 100)

        GI_envelope = np.array([find_GI_for_GII(t) for t in GII_range])

        # Remove nan values where no root was found
        valid_points = ~np.isnan(GI_envelope)
        valid_GII_range = GII_range[valid_points]
        GI_envelope = GI_envelope[valid_points]

        ax.plot(
            GI_envelope,
            valid_GII_range,
            "--",
            linewidth=2,
            label="Fracture Toughness Envelope",
            color="red",
        )

        # Formatting
        ax.set_xlabel("GI (J/m²)")
        ax.set_ylabel("GII (J/m²)")
        ax.set_title("Fracture Toughness Envelope")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.set_xlim(0, max(np.abs(GI_envelope)) * 1.1)
        ax.set_ylim(0, max(np.abs(valid_GII_range)) * 1.1)

        plt.tight_layout()

        self._save_figure(filename, fig)

        return fig

    def plot_analysis(
        self,
        system: SystemModel,
        criteria_evaluator: CriteriaEvaluator,
        min_force_result: FindMinimumForceResult,
        min_crack_length: float,
        coupled_criterion_result: CoupledCriterionResult,
        dz: int = 2,
        deformation_scale: float = 100.0,
        window: int = np.inf,
        levels: int = 300,
        filename: str = "analysis",
    ) -> Figure:
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
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)

        logger.debug("System Segments: %s", system.scenario.segments)
        analyzer = Analyzer(system)
        xsl, z, xwl = analyzer.rasterize_solution(mode="cracked", num=200)

        zi = analyzer.get_zmesh(dz=dz)["z"]
        H = analyzer.sm.slab.H
        h = system.weak_layer.h
        system_type = analyzer.sm.scenario.system_type
        fq = analyzer.sm.fq

        # Generate a window size which fits the plots
        window = min(window, np.max(xwl) - np.min(xwl), 10000)

        # Calculate scaling factors for proper aspect ratio and relative heights
        # 7:1 aspect ratio: vertical extent = window / 7
        total_vertical_extent = window / 7.0

        # Slab should appear 2x taller than weak layer
        # So slab gets 2/3 of vertical space, weak layer gets 1/3
        slab_display_height = (2 / 3) * total_vertical_extent
        weak_layer_display_height = (1 / 3) * total_vertical_extent

        # Calculate separate scaling factors for coordinates
        slab_z_scale = slab_display_height / H
        weak_layer_z_scale = weak_layer_display_height / h

        # Deformation scaling (separate from coordinate scaling)
        scale = deformation_scale

        # Compute slab displacements on grid (cm)
        Usl = np.vstack([fq.u(z, h0=h0, unit="cm") for h0 in zi])
        Wsl = np.vstack([fq.w(z, unit="cm") for _ in zi])
        Sigmawl = np.where(np.isfinite(xwl), fq.sig(z, unit="kPa"), np.nan)
        Tauwl = np.where(np.isfinite(xwl), fq.tau(z, unit="kPa"), np.nan)

        # Put coordinate origin at horizontal center
        if system_type in ["skier", "skiers"]:
            xsl = xsl - max(xsl) / 2
            xwl = xwl - max(xwl) / 2

        # Compute slab grid coordinates with vertical origin at top surface (cm)
        Xsl, Zsl = np.meshgrid(1e-1 * (xsl), 1e-1 * slab_z_scale * (zi - H / 2))

        # Get x-coordinate of maximum deflection w (cm) and derive plot limits
        xmax = np.min([np.max([Xsl, Xsl + scale * Usl]), 1e-1 * window / 2])
        xmin = np.max([np.min([Xsl, Xsl + scale * Usl]), -1e-1 * window / 2])

        # Compute weak-layer grid coordinates (cm)
        # Position weak layer below the slab
        Xwl, Zwl = np.meshgrid(
            1e-1 * xwl,
            [
                0,  # Top of weak layer (at bottom of slab)
                1e-1 * weak_layer_z_scale * h,  # Bottom of weak layer
            ],
        )

        # Assemble weak-layer displacement field (top and bottom)
        Uwl = np.vstack([Usl[-1, :], np.zeros(xwl.shape[0])])
        Wwl = np.vstack([Wsl[-1, :], np.zeros(xwl.shape[0])])

        stress_envelope = criteria_evaluator.stress_envelope(
            Sigmawl, Tauwl, system.weak_layer
        )
        stress_envelope[np.isnan(stress_envelope)] = np.nanmax(stress_envelope)

        # Assemble weak-layer output on grid
        weak = np.vstack([stress_envelope, stress_envelope])

        # Normalize colormap
        levels = np.linspace(0, 1, num=levels + 1, endpoint=True)

        # Plot outlines of the undeformed and deformed slab
        ax.plot(
            _outline(Xsl),
            _outline(Zsl),
            linestyle="--",
            color="yellow",
            alpha=0.3,
            linewidth=1,
        )
        ax.plot(
            _outline(Xsl + scale * Usl),
            _outline(Zsl + scale * Wsl),
            color="blue",
            linewidth=1,
        )

        # Plot deformed weak-layer _outline
        nanmask = np.isfinite(xwl)
        ax.plot(
            _outline(Xwl[:, nanmask] + scale * Uwl[:, nanmask]),
            _outline(Zwl[:, nanmask] + scale * Wwl[:, nanmask]),
            "k",
            linewidth=1,
        )

        cmap = plt.get_cmap("RdBu_r")
        cmap.set_over(_adjust_lightness(cmap(1.0), 0.9))
        cmap.set_under(_adjust_lightness(cmap(0.0), 0.9))

        ax.contourf(
            Xwl + scale * Uwl,
            Zwl + scale * Wwl,
            weak,
            levels=levels,
            cmap=cmap,
            extend="both",
        )

        # Plot setup
        ax.axis("scaled")
        ax.set_xlim([xmin, xmax])
        ax.invert_yaxis()
        ax.use_sticky_edges = False

        # Set up custom y-axis ticks to show real scaled heights
        # Calculate the actual extent of the plot
        slab_top = 1e-1 * slab_z_scale * (zi[0] - H / 2)  # Top of slab
        slab_bottom = 1e-1 * slab_z_scale * (zi[-1] - H / 2)  # Bottom of slab
        weak_layer_bottom = 1e-1 * weak_layer_z_scale * h  # Bottom of weak layer

        # Create tick positions and labels
        y_ticks = []
        y_labels = []

        # Slab ticks (show actual slab heights in mm)
        num_slab_ticks = 5
        slab_tick_positions = np.linspace(slab_bottom, slab_top, num_slab_ticks)
        slab_height_ticks = np.linspace(
            0, -H, num_slab_ticks
        )  # Actual slab heights in mm

        for pos, height in zip(slab_tick_positions, slab_height_ticks):
            y_ticks.append(pos)
            y_labels.append(f"{height:.0f}")

        # Weak layer ticks (show actual weak layer heights in mm)
        num_wl_ticks = 3
        wl_tick_positions = np.linspace(0, weak_layer_bottom, num_wl_ticks)
        wl_height_ticks = np.linspace(
            0, h, num_wl_ticks
        )  # Actual weak layer heights in mm

        for pos, height in zip(wl_tick_positions, wl_height_ticks):
            y_ticks.append(pos)
            y_labels.append(f"{height:.0f}")

        # Set the custom ticks
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        # Add grid lines for better readability
        ax.grid(True, alpha=0.3)

        # Add horizontal line to separate slab and weak layer
        ax.axhline(y=slab_bottom, color="black", linewidth=1, alpha=0.5, linestyle="--")

        # === ADD ANALYSIS ANNOTATIONS ===

        # 1. Vertical lines for min_crack_length (centered at x=0)
        min_crack_length_cm = min_crack_length / 10  # Convert mm to cm
        ax.plot(
            [-min_crack_length_cm / 2, -min_crack_length_cm / 2],
            [0, weak_layer_bottom],
            color="orange",
            linewidth=1,
            alpha=0.7,
            label=f"Crack Propagation: ±{min_crack_length / 2:.0f}mm",
        )
        ax.plot(
            [min_crack_length_cm / 2, min_crack_length_cm / 2],
            [0, weak_layer_bottom],
            color="orange",
            linewidth=1,
            alpha=0.7,
        )

        base_square_size = (1e-1 * window) / 25  # Base size for scaling
        segment_position = 0  # Track cumulative position
        square_spacing = 2.0  # Space above slab for squares

        # Collect weight information for legend
        weight_legend_items = []

        for segment in system.scenario.segments:
            segment_position += segment.length
            if segment.m > 0:  # If there's a weight at this segment
                # Convert position to cm and center at x=0
                square_x = (segment_position / 10) - (1e-1 * max(xsl))
                square_y = slab_top - square_spacing  # Position above slab

                # Calculate square side length based on cube root of weight (volume scaling)
                actual_side_length = base_square_size * (segment.m / 100) ** (1 / 3)

                # Draw actual skier weight square (filled, blue)
                actual_square = Rectangle(
                    (square_x - actual_side_length / 2, square_y - actual_side_length),
                    actual_side_length,
                    actual_side_length,
                    facecolor="blue",
                    alpha=0.7,
                    edgecolor="blue",
                    linewidth=1,
                )
                ax.add_patch(actual_square)

                # Add to weight legend
                weight_legend_items.append(
                    (f"Actual: {segment.m:.0f} kg", "blue", True)
                )

                # Draw critical weight square (outline only, green)
                critical_weight = min_force_result.critical_skier_weight
                critical_side_length = base_square_size * (critical_weight / 100) ** (
                    1 / 3
                )
                critical_square = Rectangle(
                    (
                        square_x - critical_side_length / 2,
                        square_y - critical_side_length,
                    ),
                    critical_side_length,
                    critical_side_length,
                    facecolor="none",
                    alpha=0.7,
                    edgecolor="green",
                    linewidth=1,
                )
                ax.add_patch(critical_square)

                # Add to weight legend (only once)
                if not any("Critical" in item[0] for item in weight_legend_items):
                    weight_legend_items.append(
                        (f"Critical: {critical_weight:.0f} kg", "green", False)
                    )

        # 3. Coupled criterion result square (centered at x=0)
        coupled_weight = coupled_criterion_result.critical_skier_weight
        coupled_side_length = base_square_size * (coupled_weight / 100) ** (1 / 3)
        coupled_square = Rectangle(
            (-coupled_side_length / 2, slab_top - square_spacing - coupled_side_length),
            coupled_side_length,
            coupled_side_length,
            facecolor="none",
            alpha=0.7,
            edgecolor="red",
            linewidth=1,
        )
        ax.add_patch(coupled_square)

        # Add to weight legend
        weight_legend_items.append((f"Coupled: {coupled_weight:.0f} kg", "red", False))

        # 4. Vertical line for coupled criterion result (spans weak layer only)
        cc_crack_length = coupled_criterion_result.crack_length / 10
        ax.plot(
            [cc_crack_length / 2, cc_crack_length / 2],
            [0, weak_layer_bottom],
            color="red",
            linewidth=1,
            alpha=0.7,
        )
        ax.plot(
            [-cc_crack_length / 2, -cc_crack_length / 2],
            [0, weak_layer_bottom],
            color="red",
            linewidth=1,
            alpha=0.7,
            label=f"Crack Nucleation: ±{coupled_criterion_result.crack_length / 2:.0f}mm",
        )

        # Calculate and set proper y-axis limits to include squares
        # Find the maximum extent of squares and text above the slab
        max_weight = max(
            [segment.m for segment in system.scenario.segments if segment.m > 0]
            + [
                min_force_result.critical_skier_weight,
                coupled_criterion_result.critical_skier_weight,
            ]
        )
        max_square_size = base_square_size * (max_weight / 100) ** (1 / 3)

        # Calculate plot limits for inverted y-axis
        # Top of plot (smallest y-value): above the squares and text
        plot_top = slab_top - 3 * max_square_size - 5  # Include text space

        # Bottom of plot (largest y-value): below weak layer
        plot_bottom = weak_layer_bottom + 1.0

        # Set y-limits [bottom, top] for inverted axis
        ax.set_ylim([plot_bottom, plot_top])

        weight_legend_handles = []
        weight_legend_labels = []

        for label, color, filled in weight_legend_items:
            if filled:
                # Filled square for actual weights
                patch = Patch(facecolor=color, edgecolor=color, alpha=0.7)
            else:
                # Outline only square for critical/coupled weights
                patch = Patch(facecolor="none", edgecolor=color, alpha=0.7, linewidth=1)

            weight_legend_handles.append(patch)
            weight_legend_labels.append(label)

        # Plot labels
        ax.set_xlabel(r"lateral position $x$ (cm) $\longrightarrow$")
        ax.set_ylabel("Layer Height (mm)\n" + r"$\longleftarrow $ Slab | Weak Layer")

        # Add primary legend for annotations (crack lengths)
        legend1 = ax.legend(loc="upper right", fontsize=8)

        # Add the first legend back (matplotlib only shows the last legend by default)
        ax.add_artist(legend1)

        # Show colorbar
        ticks = np.linspace(levels[0], levels[-1], num=11, endpoint=True)
        fig.colorbar(
            ax.contourf(
                Xwl + scale * Uwl,
                Zwl + scale * Wwl,
                weak,
                levels=levels,
                cmap=cmap,
                extend="both",
            ),
            orientation="horizontal",
            ticks=ticks,
            label="Stress Criterion: Failure > 1",
            aspect=35,
        )

        # Save figure
        self._save_figure(filename, fig)

        return fig

    # === PLOT WRAPPERS ===========================================================

    def plot_displacements(
        self,
        analyzer: Analyzer,
        x: np.ndarray,
        z: np.ndarray,
        filename: str = "displacements",
    ) -> Figure:
        """Wrap for displacements plot."""
        if not analyzer.sm.is_generalized:
            data = [
                [x / 10, analyzer.sm.fq.u(z, unit="mm"), r"$u_0\ (\mathrm{mm})$"],
                [x / 10, -analyzer.sm.fq.w(z, unit="mm"), r"$-w\ (\mathrm{mm})$"],
                [x / 10, analyzer.sm.fq.psi(z, unit="deg"), r"$\psi\ (^\circ)$ "],
                ]
        else:
            data = [
                [x / 10, analyzer.sm.fq.u(z, unit="mm"), r"$u_0\ (\mathrm{mm})$"],
                [x / 10, analyzer.sm.fq.v(z, unit="mm"), r"$v_0\ (\mathrm{mm})$"],
                [x / 10, -analyzer.sm.fq.w(z, unit="mm"), r"$-w\ (\mathrm{mm})$"],
                [x / 10, analyzer.sm.fq.psiy(z, unit="deg"), r"$\psi_y\ (^\circ)$ "],
                [x / 10, analyzer.sm.fq.psiz(z, unit="deg"), r"$\psi_z\ (^\circ)$ "],
                ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            ax1label=r"Displacements",
            ax1data=data,
            filename=filename,
        )

    def plot_stresses(
        self,
        analyzer: Analyzer,
        x: np.ndarray,
        z: np.ndarray,
        filename: str = "stresses",
    ) -> Figure:
        """Wrap stress plot."""
        if not analyzer.sm.is_generalized:
            data = [
                [x / 10, analyzer.sm.fq.tau(z, unit="kPa"), r"$\tau$"],
                [x / 10, analyzer.sm.fq.sig(z, unit="kPa"), r"$\sigma$"],
            ]
        else:
            data = [
                [x / 10, analyzer.sm.fq.tau_xz(z, unit="kPa"), r"$\tau_{xz}$"],
                [x / 10, analyzer.sm.fq.tau_yz(z, unit="kPa"), r"$\tau_{yz}$"],
                [x / 10, analyzer.sm.fq.sig_zz(z, unit="kPa"), r"$\sigma_{zz}$"],
            ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            ax1label=r"Stress (kPa)",
            ax1data=data,
            filename=filename,
        )

    def plot_stress_criteria(
        self, analyzer: Analyzer, x: np.ndarray, stress: np.ndarray
    ) -> Figure:
        """Wrap plot of stress and energy criteria."""
        data = [[x / 10, stress, r"$\sigma/\sigma_\mathrm{c}$"]]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            ax1label=r"Criteria",
            ax1data=data,
            filename="crit",
        )

    def plot_ERR_comp(
        self,
        analyzer: Analyzer,
        da: np.ndarray,
        Gdif: np.ndarray,
        Ginc: np.ndarray,
        mode: int = 0,
    ) -> Figure:
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
            filename="err",
            vlines=False,
        )

    def plot_ERR_modes(
        self, analyzer: Analyzer, da: np.ndarray, G: np.ndarray, kind: str = "inc"
    ) -> Figure:
        """Wrap energy release rate plot."""
        label = r"$\bar{\mathcal{G}}$" if kind == "inc" else r"$\mathcal{G}$"
        data = [
            [da / 10, 1e3 * G[3, :], label + r"$_\mathrm{I\!I\!I}$"],
            [da / 10, 1e3 * G[2, :], label + r"$_\mathrm{I\!I}$"],
            [da / 10, 1e3 * G[1, :], label + r"$_\mathrm{I}$"],
            [da / 10, 1e3 * G[0, :], label + r"$_\mathrm{I+I\!I+I\!I\!I}$"],
        ]
        self._plot_data(
            scenario=analyzer.sm.scenario,
            xlabel=r"Crack length $a$ (cm)",
            ax1label=r"Energy release rate (J/m$^2$)",
            ax1data=data,
            filename="modes",
            vlines=False,
        )

    def plot_fea_disp(
        self, analyzer: Analyzer, x: np.ndarray, z: np.ndarray, fea: np.ndarray
    ) -> Figure:
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
            filename="fea_disp",
            labelpos=-50,
        )

    def plot_fea_stress(
        self, analyzer: Analyzer, xb: np.ndarray, zb: np.ndarray, fea: np.ndarray
    ) -> Figure:
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
            filename="fea_stress",
            labelpos=-50,
        )

    # === BASE PLOT FUNCTION ======================================================

    def _plot_data(
        self,
        scenario: Scenario,
        filename: str,
        ax1data,
        ax1label,
        ax2data=None,
        ax2label=None,
        labelpos=None,
        vlines=True,
        xlabel=r"Horizontal position $x$ (cm)",
    ) -> Figure:
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
            labelpos = int(0.5 * len(x[~np.isnan(x)]))

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
        if filename:
            self._save_figure(filename, fig)

        return fig
