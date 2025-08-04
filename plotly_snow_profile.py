### SnowProfile
import copy
from typing import Literal
from itertools import groupby

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from weac_2.components import WeakLayer, Layer


def snow_profile(weaklayer: WeakLayer, layers: list[Layer]):
    """
    Generates a snow stratification profile plot using Plotly.

    Parameters:
    - weaklayer_thickness (float): Thickness of the weak layer in the snowpack.
    - layers (list of dicts): Each dict has keys density, thickness, hardness, and grain of a layer.

    Returns:
    - fig (go.Figure): A Plotly figure object representing the snow profile.
    """
    # Define colors
    COLORS = {
        "slab_fill": "#9ec1df",
        "slab_line": "rgba(4, 110, 124, 0.812)",
        "weak_layer_fill": "#E57373",
        "weak_layer_line": "#FFCDD2",
        "weak_layer_text": "#FFCDD2",
        "substratum_fill": "#607D8B",
        "substratum_line": "#ECEFF1",
        "substratum_text": "#ECEFF1",
        "background": "rgb(134, 148, 160)",
        "lines": "rgb(134, 148, 160)",
    }

    # reverse layers
    layers = copy.deepcopy(layers)

    # Compute total height and set y-axis maximum
    total_height = sum(layer.h for layer in layers)
    y_max = max(total_height, 450)  # Ensure y_max is at least 450

    # Compute x-axis maximum based on layer densities
    max_density = max((layer.rho for layer in layers), default=400)
    x_max = max(1.05 * max_density, 300)  # Ensure x_max is at least 300

    # Initialize the Plotly figure
    fig = go.Figure()

    # Initialize variables for plotting layers
    previous_density = 0  # Start from zero density
    previous_height = 0

    # Define positions for annotations (table columns)
    col_width = 0.12
    col_width = min(col_width * x_max, 30)
    x_pos = {
        "col0_start": 0 * col_width,
        "col1_start": 1 * col_width,
        "col2_start": 2 * col_width,
        "col3_start": 3 * col_width,
        "col3_end": 4 * col_width,
    }

    # Compute midpoints for annotation placement
    first_column_mid = (x_pos["col0_start"] + x_pos["col1_start"]) / 2
    second_column_mid = (x_pos["col1_start"] + x_pos["col2_start"]) / 2
    third_column_mid = (x_pos["col2_start"] + x_pos["col3_start"]) / 2
    fourth_column_mid = (x_pos["col3_start"] + x_pos["col3_end"]) / 2

    # Calculate average height per table row
    num_layers = max(len(layers), 1)
    min_table_row_height = (y_max / 2) / num_layers
    max_table_row_height = 300
    avg_row_height = (y_max) / num_layers
    avg_row_height = min(avg_row_height, max_table_row_height)
    avg_row_height = max(avg_row_height, min_table_row_height)
    # Taken space for the table
    table_height = avg_row_height * num_layers
    table_offset = total_height - table_height

    # Initialize current table height
    current_height = 0
    current_table_y = table_offset

    # Loop through each layer and plot
    for layer in layers:
        density = layer.rho
        thickness = layer.h
        hand_hardness = layer.hand_hardness
        grain = layer.grain_type

        # Define layer boundaries
        layer_bottom = current_height
        layer_top = current_height + thickness

        # Plot the layer
        fig.add_shape(
            type="rect",
            x0=-density,
            x1=0,
            y0=layer_bottom,
            y1=layer_top,
            fillcolor=COLORS["slab_fill"],
            line=dict(width=0.4, color=COLORS["slab_fill"]),
            layer="above",
        )

        # Plot lines connecting previous and current densities
        fig.add_shape(
            type="line",
            x0=-previous_density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_bottom,
            line=dict(color=COLORS["slab_line"], width=1.2),
        )
        fig.add_shape(
            type="line",
            x0=-density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_top,
            line=dict(color=COLORS["slab_line"], width=1.2),
        )

        # Add heights on the right of layer changes
        fig.add_annotation(
            x=first_column_mid,
            y=layer_bottom,
            text=str(round(layer_bottom)),
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )

        # Define table row boundaries
        table_bottom = current_table_y
        table_top = current_table_y + avg_row_height

        # Add table grid lines
        fig.add_shape(
            type="line",
            x0=x_pos["col1_start"],
            y0=table_bottom,
            x1=x_pos["col3_end"],
            y1=table_bottom,
            line=dict(color="lightgrey", width=0.5),
        )

        # Add annotations for density, grain form, and hand hardness
        fig.add_annotation(
            x=second_column_mid,
            y=(table_bottom + table_top) / 2,
            text=str(round(density)),
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=third_column_mid,
            y=(table_bottom + table_top) / 2,
            text=grain if grain else "-",
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=fourth_column_mid,
            y=(table_bottom + table_top) / 2,
            text=hand_hardness if hand_hardness else "-",
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )

        # Lines from layer edges to table
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_top,
            x1=x_pos["col1_start"],
            y1=table_top,
            line=dict(color="lightgrey", width=0.5),
        )

        # Update variables for next iteration
        previous_density = density
        current_height = layer_top
        current_table_y = table_top

    # Additional cases which are not covered by the loop
    print(previous_density)
    # Additional case: Add density line from last layer to x=0
    fig.add_shape(
        type="line",
        x0=-previous_density,
        y0=total_height,
        x1=0.0,
        y1=total_height,
        line=dict(width=1.2, color=COLORS["slab_line"]),
    )
    # Additional case: Add table grid of last layer
    fig.add_shape(
        type="line",
        x0=x_pos["col1_start"],
        y0=total_height,
        x1=x_pos["col3_end"],
        y1=total_height,
        line=dict(color="lightgrey", width=0.5),
    )
    # Additional case: Add layer edge line from first layer to table
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=x_pos["col1_start"],
        y1=table_offset,
        line=dict(width=0.5, color="lightgrey"),
    )

    fig.add_annotation(
        x=x_pos["col0_start"],
        y=total_height,
        text=str(round(0)),
        showarrow=False,
        font=dict(size=10),
        xanchor="left",
        yanchor="middle",
    )

    # Vertical lines for table columns
    for x in [
        x_pos["col1_start"],
        x_pos["col2_start"],
        x_pos["col3_start"],
    ]:
        fig.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=y_max,
            line=dict(color="lightgrey", width=0.5),
        )

    column_header_y = -200
    # Horizontal line at table header
    fig.add_shape(
        type="line",
        x0=0,
        y0=column_header_y,
        x1=x_pos["col3_end"],
        y1=column_header_y,
        line=dict(color="lightgrey", width=0.5),
    )

    # Annotations for table headers
    header_y_position = (column_header_y) / 2
    fig.add_annotation(
        x=first_column_mid,
        y=header_y_position,
        text="H",  # "H<br>cm",  # "H (cm)",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=second_column_mid,
        y=header_y_position,
        text="D",  # 'D<br>kg/m³',  # "Density (kg/m³)",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=third_column_mid,
        y=header_y_position,
        text="F",  # "GF",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=fourth_column_mid,
        y=header_y_position,
        text="R",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )

    fig.add_annotation(
        x=0.0,
        y=-0.06,
        text="H: Height (cm)  D: Density (kg/m³)  F: Grain Form  R: Hand Hardness",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=10),
        align="left",
    )

    # Set axes properties
    fig.update_layout(
        xaxis=dict(
            range=[-1.05 * x_max, x_pos["col3_end"]],
            autorange=False,
            tickvals=[-400, -300, -200, -100, 0],
            ticktext=["400", "300", "200", "100", "0"],
        ),
        yaxis=dict(
            range=[total_height, -200.0],
            domain=[0.0, 1.0],
            # showgrid=True,
            # gridcolor="lightgray",
            # gridwidth=1,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            showticklabels=False,
            # tickmode="linear",
            # tick0=0,
            # dtick=max(total_height * 0.2, 10),  # Tick every 50 units
            # tickcolor="black",
            # tickwidth=2,
            # ticklen=5,
        ),
        height=600,
        width=600,
        margin=dict(l=0, r=0, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def criticality_plots(
    weaklayer: WeakLayer, layers: list[Layer], dataframe: pd.DataFrame
):
    fig = go.Figure()

    # Extract cirtical values.
    critical_cc = 100.0
    critical_sserr = 30.0
    depth = max(dataframe["wl_depth"])

    # Extract highest values
    max_sserr = max(dataframe["sserr_result"])
    max_cc = max(dataframe["coupled_criterion"])
    # Extract lowest values
    min_sserr = min(dataframe["sserr_result"])
    min_cc = min(dataframe["coupled_criterion"])

    # Append 0.0 depth to dataframe
    dataframe = pd.concat(
        [
            dataframe,
            pd.DataFrame(
                {
                    "wl_depth": [0.0],
                    "sserr_result": [0.0],
                    "coupled_criterion": [min_cc],
                }
            ),
        ]
    )
    dataframe = dataframe.sort_values(by="wl_depth")

    # Interpolate 1D densely: x10 resolution
    y_depths = np.linspace(0, depth, 10 * len(dataframe))
    x_sserr = np.interp(y_depths, dataframe["wl_depth"], dataframe["sserr_result"])
    x_cc = np.interp(y_depths, dataframe["wl_depth"], dataframe["coupled_criterion"])

    # Extract region where cc is self-collapsed
    cc_zero_mask = x_cc <= 1e-6

    # Robustify division
    epsilon = 1e-6
    x_cc = np.where(cc_zero_mask, epsilon, x_cc)

    x_sserr = x_sserr / critical_sserr
    x_cc = critical_cc / x_cc

    # Define colors for each axis
    AXIS_COLORS = {
        "sserr": "blue",
        "cc": "orange",
    }

    fig.add_trace(
        go.Scatter(
            x=x_sserr,
            y=y_depths,
            mode="lines",
            name="Energy Release Rate",
            line=dict(color=AXIS_COLORS["sserr"], width=3),
            marker=dict(size=6, color=AXIS_COLORS["sserr"]),
            xaxis="x1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_cc,
            y=y_depths,
            mode="lines",
            name="Critical Coupling",
            line=dict(color=AXIS_COLORS["cc"], width=3),
            marker=dict(size=6, color=AXIS_COLORS["cc"]),
            xaxis="x1",
        )
    )
    # fig.add_vline(x=1.0, line=dict(color="black", width=3))
    fig.add_trace(
        go.Scatter(
            x=[1.0, 1.0],
            y=[0.0, depth],
            mode="lines",
            name="Critical Point",
            line=dict(color="black", width=2),
            showlegend=False,  # optional
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[0.0],
            mode="markers",
            name="Critical Point",
            marker=dict(size=10, color="black"),
            showlegend=False,  # optional
        )
    )

    # Create points for filled region between x_vals and x=1.0
    x_shading = np.concatenate(
        [
            x_sserr,
            np.full_like(x_sserr, 1.0)[::-1],
        ]
    )
    y_shading = np.concatenate([y_depths, y_depths[::-1]])
    above_mask = x_shading >= 1.0

    segments = []
    for is_above, group in groupby(enumerate(above_mask), lambda x: x[1]):
        if is_above:
            indices = [i for i, _ in group]
            segments.append(indices)

    for segment in segments:
        # only keep points where x_shading is >= 1.0
        plot_x = x_shading[segment]
        plot_y = y_shading[segment]

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.2)",  # blue-ish transparent
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Shaded Criticality",
            )
        )

    # Create points for filled region between x_vals and x=1.0
    x_shading = x_cc[~cc_zero_mask]
    y_shading = y_depths[~cc_zero_mask]
    above_mask = x_shading >= 1.0

    segments = []
    for is_above, group in groupby(enumerate(above_mask), lambda x: x[1]):
        if is_above:
            indices = [i for i, _ in group]
            segments.append(indices)

    for segment in segments:
        # only keep points where x_shading is >= 1.0
        plot_x = np.concatenate(
            [
                x_shading[segment],
                np.full_like(x_shading[segment], 1.0)[::-1],
            ]
        )
        plot_y = np.concatenate([y_shading[segment], y_shading[segment][::-1]])

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                fill="toself",
                fillcolor="rgba(255, 165, 0, 0.2)",  # orange-ish transparent
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Shaded Criticality",
            )
        )

    # Create self-collapsed region
    x_shading = x_cc
    y_shading = y_depths
    segments = []
    for is_above, group in groupby(enumerate(cc_zero_mask), lambda x: x[1]):
        if is_above:
            indices = [i for i, _ in group]
            segments.append(indices)

    for segment in segments:
        # only keep points where x_shading is >= 1.0
        plot_x = np.concatenate(
            [
                x_shading[segment],
                np.full_like(x_shading[segment], 1.0)[::-1],
            ]
        )
        plot_y = np.concatenate([y_shading[segment], y_shading[segment][::-1]])

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                fill="toself",
                fillcolor="rgba(0, 0, 0, 0.1)",  # light-grey
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Self-Collapsed",
            )
        )

    # Configure multiple overlaying x-axes with enhanced colors and ticks
    fig.update_layout(
        # Main y-axis
        yaxis=dict(
            title="Depth [mm]",  # Remove built-in title, we'll use annotation
            range=[depth, -200.0],
            domain=[0.0, 1.0],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=2,
            tickmode="linear",
            tick0=0,
            dtick=max(depth * 0.2, 10),  # Tick every 50 units
            tickcolor="black",
            tickwidth=2,
            ticklen=5,
        ),
        # First x-axis (SSERR) - primary axis
        xaxis=dict(
            title="",  # Remove built-in title, we'll use annotation
            range=[0, 2.0],
            side="bottom",
            # autorange="reversed",
            showgrid=True,
            gridcolor="lightblue",
            gridwidth=1,
            tickmode="linear",
            tick0=0,
            dtick=2.0 * 0.1,  # 5 ticks across the range
            tickcolor="black",
            tickwidth=2,
            ticklen=8,
            tickfont=dict(color="black", size=10),
            linecolor="black",
            linewidth=2,
        ),
        # # Second x-axis (Coupled Criterion)
        # xaxis2=dict(
        #     title="",  # Remove built-in title, we'll use annotation
        #     range=[0.0, 2.0],
        #     anchor="free",
        #     overlaying="x",
        #     side="bottom",
        #     position=0.05,
        #     zeroline=True,
        #     zerolinecolor=AXIS_COLORS["cc"],
        #     zerolinewidth=2,
        #     showgrid=False,  # Avoid grid overlap
        #     tickmode="linear",
        #     # autorange="reversed",
        #     tick0=0,
        #     dtick=2.0 * 0.2,  # 5 ticks across the range
        #     tickcolor=AXIS_COLORS["cc"],
        #     tickwidth=2,
        #     ticklen=8,
        #     tickfont=dict(color=AXIS_COLORS["cc"], size=10),
        #     linecolor=AXIS_COLORS["cc"],
        #     linewidth=2,
        # ),
        showlegend=False,
        # legend=dict(
        #     x=1.02,
        #     y=1,
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor="black",
        #     borderwidth=1,
        # ),
        width=400,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=40),
    )

    # X-axis title annotations positioned above their respective axes
    fig.add_annotation(
        text="Criticality",
        x=0.5,  # Center of the plot
        y=0.0,  # Just above the bottom axis
        xref="paper",
        yref="paper",
        ax=0,
        ay=20,
        font=dict(size=12),
    )

    fig.add_annotation(
        text="Critical Point",
        x=0.5,
        y=1.0,
        xref="paper",
        yref="paper",
        ax=0,  # Shift text 40px right
        ay=-10,
        font=dict(color="black"),
    )
    return fig


def criticality_heatmap(
    weaklayer: WeakLayer, layers: list[Layer], dataframe: pd.DataFrame
):
    # Parameters
    critical_cc = 100.0
    critical_sserr = 30.0

    # Get max depth
    depth = max(dataframe["wl_depth"])

    # Extend dataframe with 0-depth row if not already present
    if not (dataframe["wl_depth"] == 0.0).any():
        dataframe = pd.concat(
            [
                dataframe,
                pd.DataFrame(
                    {
                        "wl_depth": [0.0],
                        "sserr_result": [0.0],
                        "coupled_criterion": [dataframe["coupled_criterion"].min()],
                    }
                ),
            ]
        )

    dataframe = dataframe.sort_values(by="wl_depth")

    # Interpolate: y = depth in cm (or mm depending on your unit)
    y_depths = np.linspace(0, depth, 10 * len(dataframe))
    x_sserr = np.interp(y_depths, dataframe["wl_depth"], dataframe["sserr_result"])
    x_cc = np.interp(y_depths, dataframe["wl_depth"], dataframe["coupled_criterion"])

    # Extract region where cc is self-collapsed
    cc_zero_mask = x_cc <= 1e-6

    # Avoid division by zero
    epsilon = 1e-6
    x_cc = np.where(x_cc <= epsilon, epsilon, x_cc)

    # Normalize
    x_sserr /= critical_sserr
    x_sserr = np.clip(x_sserr, 0.0, 1.0)  # Limit max to 1.0
    x_cc = critical_cc / x_cc
    x_cc = np.clip(x_cc, 0.0, 1.0)  # Limit max to 1.0
    x_cc[cc_zero_mask] = 0.0

    # Create 2D z-values for heatmap (duplicate along x-axis)
    z_cc = np.tile(x_cc.reshape(-1, 1), (1, 2))  # Shape: (len(y_depths), 2)
    x_vals = [0.0, 0.5, 1.0]
    z_sserr = np.tile(x_sserr.reshape(-1, 1), (1, 2))  # Shape: (len(y_depths), 2)
    x_vals_2 = [1.0, 1.5, 2.0]

    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z_cc,
            x=x_vals,
            y=y_depths,
            colorscale="Reds",
            showscale=False,
            reversescale=False,
            zmin=0.0,
            zmax=1.0,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=z_sserr,
            x=x_vals_2,
            y=y_depths,
            colorscale="Reds",
            showscale=False,
            reversescale=False,
            zmin=0.0,
            zmax=1.0,
            hoverinfo="skip",
        )
    )

    # Create a scaling between the two heatmaps
    z_combined = z_cc * 0.35 + z_sserr * 0.75
    z_combined = np.where(z_cc == 0.0, 0.0, z_combined)
    z_combined = np.where(z_sserr == 0.0, 0.0, z_combined)
    z_combined = np.clip(z_combined, 0.0, 1.0)
    x_vals_3 = [2.0, 2.5, 3.0]

    # traffic_light_fade = [
    #     [0.00, "rgb(0,180,0)"],  # green
    #     [0.10, "rgb(80,200,0)"],  # lighter green
    #     [0.20, "rgb(170,220,0)"],  # yellow-green
    #     [0.33, "yellow"],  # yellow
    #     [0.45, "rgb(255,180,0)"],  # yellow-orange
    #     [0.55, "orange"],  # orange
    #     [0.70, "orangered"],  # deep orange
    #     [0.85, "red"],
    #     [1.00, "darkred"],
    # ]
    twilight_fade = [
        [0.00, "rgb(20,30,80)"],  # deep indigo / night sky
        [0.15, "rgb(60,50,150)"],  # violet
        [0.30, "rgb(120,60,200)"],  # magenta
        [0.45, "rgb(200,90,220)"],  # soft pink-violet
        [0.60, "rgb(255,140,180)"],  # pink-orange
        [0.75, "rgb(255,180,120)"],  # warm peach
        [0.90, "rgb(255,210,100)"],  # sunset orange
        [1.00, "rgb(255,240,150)"],  # fading gold
    ]

    fig.add_trace(
        go.Heatmap(
            z=z_combined,
            x=x_vals_3,
            y=y_depths,
            colorscale=twilight_fade[::-1],
            showscale=True,
            colorbar=dict(title="Cum."),
            zmin=0.0,
            zmax=1.0,
        )
    )

    xs = [2.0, 2.3, 2.6, 2.9]
    for x in xs:
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[0, depth],
                mode="lines",
                line=dict(color="lightgrey", width=0.5),
                showlegend=False,
            )
        )

    # Manual horizontal grid lines (y-direction)
    y_step = 50  # or however you want to space the grid
    y_grid = np.arange(0, depth + y_step, y_step)

    for y in y_grid:
        fig.add_trace(
            go.Scatter(
                x=[0.0, 3.0],
                y=[y, y],
                mode="lines",
                line=dict(color="white", width=0.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    xs = z_combined.mean(axis=1) + 2.0
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_depths,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

    fig.update_layout(
        yaxis=dict(
            autorange=False,
            range=[depth, -200.0],
            domain=[0.0, 1.0],
            # showgrid=False,
            # gridcolor="white",
            # gridwidth=1,
            # tickmode="linear",
            # tick0=0,
            # dtick=max(depth * 0.2, 10),  # Tick every 50 units
            # tickcolor="black",
            # tickwidth=2,
            # ticklen=5,
            showticklabels=False,
            # layer="above traces",
        ),
        xaxis=dict(
            range=[0.0, 3.0],
            tickvals=[0.5, 1.5, 2.0, 2.3, 2.6, 2.9],
            ticktext=[
                "Fracture",
                "Propagation",
                "0.0",
                "0.3",
                "0.6",
                "0.9",
            ],
        ),
        width=300,
        height=600,
        margin=dict(l=0, r=0, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig
