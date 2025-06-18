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
from weac_2.core.system_model import SystemModel
from weac_2.utils import isnotebook


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

        # Set up colors
        if colors is None:
            # Generate distinct colors using HSV color space
            self.colors = self._generate_colors(self.n_systems)
        else:
            if len(colors) != self.n_systems:
                raise ValueError(
                    f"Number of colors ({len(colors)}) must match number of systems ({self.n_systems})"
                )
            self.colors = colors

        # Set up plot directory
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        # Set up matplotlib style
        self._setup_matplotlib_style()

        # Cache analyzers for performance
        self._analyzers = {}

    def _generate_colors(self, n: int) -> List[str]:
        """Generate n distinct colors using HSV color space."""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 2)  # Alternate between 0.7 and 1.0
            value = 0.8 + 0.2 * ((i + 1) % 2)  # Alternate between 0.8 and 1.0
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(
                f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
            )
        return colors

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

    def _get_labels_and_colors(
        self, systems_to_plot: List[SystemModel]
    ) -> tuple[List[str], List[str]]:
        """Get corresponding labels and colors for systems to plot."""
        if systems_to_plot == self.systems:
            return self.labels, self.colors

        # Find indices of systems to plot
        labels = []
        colors = []
        for system in systems_to_plot:
            try:
                idx = self.systems.index(system)
                labels.append(self.labels[idx])
                colors.append(self.colors[idx])
            except ValueError:
                # System not in original list, use defaults
                labels.append(f"System {len(labels) + 1}")
                colors.append(self._generate_colors(1)[0])

        return labels, colors

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
        labels, colors = self._get_labels_and_colors(systems_to_plot)

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

        return ax1

    def plot_displacements(
        self,
        system_model: Optional[SystemModel] = None,
        system_models: Optional[List[SystemModel]] = None,
        filename: Optional[str] = None,
    ):
        """
        Plot displacement fields (u, w, ψ) for comparison.

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
        labels, colors = self._get_labels_and_colors(systems_to_plot)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        for system, label, color in zip(systems_to_plot, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            fq = system.fq

            # Convert x to meters for plotting
            x_m = x / 1000

            # Plot horizontal displacement u at mid-height
            u = fq.u(z, h0=0, unit="mm")
            axes[0].plot(x_m, u, color=color, label=label, linewidth=2)

            # Plot vertical displacement w
            w = fq.w(z, unit="mm")
            axes[1].plot(x_m, w, color=color, label=label, linewidth=2)

            # Plot rotation ψ
            psi = fq.psi(z, unit="deg")
            axes[2].plot(x_m, psi, color=color, label=label, linewidth=2)

        # Formatting
        axes[0].set_ylabel("u (mm)")
        axes[0].set_title("Horizontal Displacement")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("w (mm)")
        axes[1].set_title("Vertical Displacement")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel("Distance (m)")
        axes[2].set_ylabel("ψ (°)")
        axes[2].set_title("Cross-section Rotation")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            self._save_figure(filename, fig)

        return fig

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
        labels, colors = self._get_labels_and_colors(systems_to_plot)

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

    def plot_stresses(
        self,
        system_model: Optional[SystemModel] = None,
        system_models: Optional[List[SystemModel]] = None,
        filename: Optional[str] = None,
    ):
        """
        Plot weak layer stresses (σ, τ) for comparison.

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
        labels, colors = self._get_labels_and_colors(systems_to_plot)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        for system, label, color in zip(systems_to_plot, labels, colors):
            analyzer = self._get_analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            fq = system.fq

            # Convert x to meters for plotting
            x_m = x / 1000

            # Plot normal stress σ
            sigma = fq.sig(z, unit="kPa")
            axes[0].plot(x_m, sigma, color=color, label=label, linewidth=2)

            # Plot shear stress τ
            tau = fq.tau(z, unit="kPa")
            axes[1].plot(x_m, tau, color=color, label=label, linewidth=2)

        # Formatting
        axes[0].set_ylabel("σ (kPa)")
        axes[0].set_title("Weak Layer Normal Stress")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Distance (m)")
        axes[1].set_ylabel("τ (kPa)")
        axes[1].set_title("Weak Layer Shear Stress")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

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
        labels, colors = self._get_labels_and_colors(systems_to_plot)

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
        field: Literal["w", "u", "principal", "sigma", "tau"] = "w",
        system_model: Optional[SystemModel] = None,
        filename: Optional[str] = None,
        contour_levels: int = 20,
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
        contour_levels : int, default 20
            Number of contour levels
        """
        if system_model is None:
            system_model = self.systems[0]

        analyzer = self._get_analyzer(system_model)
        x, z, _ = analyzer.rasterize_solution()
        fq = system_model.fq

        # Convert coordinates
        x_m = x / 1000

        # Create mesh for contour plotting
        slab_height = system_model.slab.H / 1000  # Convert to meters
        y = np.linspace(0, slab_height, 50)
        X, Y = np.meshgrid(x_m, y)

        # Calculate field values
        if field == "w":
            field_values = fq.w(z, unit="mm")
            field_label = "Vertical Displacement w (mm)"
            cmap = "RdBu_r"
        elif field == "u":
            field_values = fq.u(z, h0=slab_height * 500, unit="mm")  # At mid-height
            field_label = "Horizontal Displacement u (mm)"
            cmap = "RdBu_r"
        elif field == "principal":
            # Calculate principal stress (simplified)
            sigma = fq.sig(z, unit="kPa")
            tau = fq.tau(z, unit="kPa")
            field_values = np.sqrt(sigma**2 + 4 * tau**2)
            field_label = "Principal Stress (kPa)"
            cmap = "plasma"
        elif field == "sigma":
            field_values = fq.sig(z, unit="kPa")
            field_label = "Normal Stress σ (kPa)"
            cmap = "RdBu_r"
        elif field == "tau":
            field_values = fq.tau(z, unit="kPa")
            field_label = "Shear Stress τ (kPa)"
            cmap = "RdBu_r"

        # Create field mesh (simplified - constant across height)
        Z = np.tile(field_values, (len(y), 1))

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot contours
        if field in ["sigma", "tau", "u", "w"]:
            # Use symmetric colormap for stress/displacement
            vmax = np.max(np.abs(field_values))
            norm = MidpointNormalize(vmin=-vmax, vmax=vmax, midpoint=0)
            contour = ax.contourf(X, Y, Z, levels=contour_levels, cmap=cmap, norm=norm)
        else:
            contour = ax.contourf(X, Y, Z, levels=contour_levels, cmap=cmap)

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(field_label)

        # Plot deformed shape (exaggerated)
        if field in ["w", "u"]:
            scale_factor = 0.1  # Exaggeration factor
            if field == "w":
                deformation = fq.w(z, unit="mm") * scale_factor / 1000
            else:
                deformation = (
                    fq.u(z, h0=slab_height * 500, unit="mm") * scale_factor / 1000
                )

            # Plot original and deformed profiles
            ax.plot(
                x_m, np.zeros_like(x_m), "k--", linewidth=1, alpha=0.5, label="Original"
            )
            ax.plot(
                x_m, deformation, "k-", linewidth=2, label=f"Deformed ({scale_factor}x)"
            )
            ax.legend()

        # Formatting
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Height (m)")
        ax.set_title(f"Deformed Slab - {field_label}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            self._save_figure(filename, fig)

        return fig

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

        labels, colors = self._get_labels_and_colors(system_models)

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
