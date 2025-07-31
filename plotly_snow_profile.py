### SnowProfile
from typing import Literal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from weac_2.components import WeakLayer, Layer
import pandas as pd
import numpy as np


def snow_profile(weaklayer: WeakLayer, layers: list[Layer], dataframe: pd.DataFrame):
    """
    Generates a snow stratification profile plot using Plotly.

    Parameters:
    - weaklayer: weaklayer
    - layers: list of layers

    Returns:
    - fig (go.Figure): A Plotly figure object representing the snow profile.
    """

    # Define colors
    COLORS = {
        "slab_fill": "#A5C9D4",  # Lighter blue
        "slab_line": "#D3EBEE",
        "weak_layer_fill": "#E57373",
        "weak_layer_line": "#FFCDD2",
        "weak_layer_text": "#FFCDD2",
        "substratum_fill": "#607D8B",
        "substratum_line": "#ECEFF1",
        "substratum_text": "#ECEFF1",
        "background": "#000000",
        "lines": "#FF0000",
    }

    # Extract params
    weak_density = weaklayer.rho
    weaklayer_thickness = weaklayer.h

    # Define substratum properties
    substratum_thickness = 50

    y_vals = dataframe["wl_depth"]
    y_vals = y_vals[::-1]
    ss_values = -dataframe["sserr_result"]  # Negative direction
    td_values = -dataframe["touchdown_distance"]
    impact_values = -dataframe["impact_criterion"]
    coupled_values = -dataframe["coupled_criterion"]

    x_max_sserr = max(-ss_values)
    x_max_td = max(-td_values)
    x_max_impact = max(-impact_values)
    x_max_coupled = max(-coupled_values)

    # Turn layers around
    layers = layers[::-1]

    # Compute total height and set y-axis maximum
    total_height = weaklayer_thickness + sum(layer.h for layer in layers)
    y_max = max(total_height * 1.1, 450)  # Ensure y_max is at least 500

    # Compute x-axis maximum based on layer densities
    max_density = max((layer.rho for layer in layers), default=400)
    x_max = max(1.05 * max_density, 400)  # Ensure x_max is at least 400

    # Initialize the Plotly figure
    fig = go.Figure()

    # Initialize variables for plotting layers
    current_height = weaklayer_thickness
    previous_density = 0  # Start from zero density

    # Define positions for annotations (table columns)
    col_width = 0.08
    x_pos = {
        "col1_start": 1 * col_width * x_max,
        "col2_start": 2 * col_width * x_max,
        "col3_start": 3 * col_width * x_max,
        "col3_end": 4 * col_width * x_max,
    }

    # Compute midpoints for annotation placement
    first_column_mid = (x_pos["col1_start"] + x_pos["col2_start"]) / 2
    second_column_mid = (x_pos["col2_start"] + x_pos["col3_start"]) / 2
    third_column_mid = (x_pos["col3_start"] + x_pos["col3_end"]) / 2

    # Set the position for the table header
    column_header_y = y_max / 1.1
    max_table_row_height = 85  # Maximum height for table rows

    # Calculate average height per table row
    num_layers = max(len(layers), 1)
    avg_row_height = (column_header_y - weaklayer_thickness) / num_layers
    avg_row_height = min(avg_row_height, max_table_row_height)

    # Initialize current table height
    current_table_y = weaklayer_thickness

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
            layer="below",
        )

        # Plot lines connecting previous and current densities
        fig.add_shape(
            type="line",
            x0=-previous_density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_bottom,
            line=dict(color=COLORS["slab_line"], width=1.2),
            layer="below",
        )
        fig.add_shape(
            type="line",
            x0=-density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_top,
            line=dict(color=COLORS["slab_line"], width=1.2),
            layer="below",
        )

        # Add height markers on the left
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_bottom,
            x1=10,
            y1=layer_bottom,
            line=dict(width=0.5, color=COLORS["lines"]),
            layer="below",
        )
        fig.add_annotation(
            x=12,
            y=layer_bottom,
            text=str(round(layer_bottom / 10)),
            showarrow=False,
            font=dict(size=10),
            xanchor="left",
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
            layer="below",
        )

        # Add annotations for density, grain form, and hand hardness
        fig.add_annotation(
            x=first_column_mid,
            y=(table_bottom + table_top) / 2,
            text=str(round(density)),
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=second_column_mid,
            y=(table_bottom + table_top) / 2,
            text=grain,
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=third_column_mid,
            y=(table_bottom + table_top) / 2,
            text=hand_hardness,
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )

        # Lines from layer edges to table
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_bottom,
            x1=x_pos["col1_start"],
            y1=table_bottom,
            line=dict(color="lightgrey", width=0.5),
            layer="below",
        )

        # Update variables for next iteration
        previous_density = density
        current_height = layer_top
        current_table_y = table_top

    # Overlay data over layers
    fig.add_trace(
        go.Scatter(
            x=ss_values,
            y=y_vals,
            mode="lines",
            name="SSERR",
            line=dict(color="red", width=2),
            marker=dict(size=4),
            yaxis="y",
            xaxis="x2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=td_values,
            y=y_vals,
            mode="lines",
            name="Touchdown Distance",
            line=dict(color="red", width=2),
            marker=dict(size=4),
            yaxis="y",
            xaxis="x3",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=impact_values,
            y=y_vals,
            mode="lines",
            name="Impact Criterion",
            line=dict(color="red", width=2),
            marker=dict(size=4),
            yaxis="y",
            xaxis="x4",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coupled_values,
            y=y_vals,
            mode="lines",
            name="Coupled Criterion",
            line=dict(color="red", width=2),
            marker=dict(size=4),
            yaxis="y",
            xaxis="x4",
        )
    )

    # Add top layer height marker
    fig.add_shape(
        type="line",
        x0=0,
        y0=total_height,
        x1=10,
        y1=total_height,
        line=dict(width=0.5, color=COLORS["lines"]),
        layer="below",
    )
    fig.add_annotation(
        x=12,
        y=total_height,
        text=str(round(total_height / 10)),
        showarrow=False,
        font=dict(size=10),
        xanchor="left",
        yanchor="middle",
    )

    # Final line connecting last density to x=0 at total_height
    fig.add_shape(
        type="line",
        x0=-previous_density,
        y0=total_height,
        x1=0,
        y1=total_height,
        line=dict(color=COLORS["slab_line"], width=1),
        layer="below",
    )

    # Set axes properties
    fig.update_layout(
        yaxis=dict(range=[-1.05 * substratum_thickness, y_max]),
        xaxis=dict(
            range=[-1.05 * x_max, x_pos["col3_end"]],
            autorange=False,
        ),
        xaxis2=dict(  # For SSERR
            # title="SSERR [J/m^2]",
            range=[1.05 * x_max_sserr, x_pos["col3_end"]],
            autorange=False,
        ),
        xaxis3=dict(  # For Touchdown Distance
            # title="Touchdown Distance [mm]",
            range=[1.05 * x_max_td, x_pos["col3_end"]],
            autorange=False,
        ),
        xaxis4=dict(  # For Impact Criterion
            # title="Criticial Weights [kg]",
            range=[1.05 * x_max_coupled, x_pos["col3_end"]],
            autorange=False,
        ),
        showlegend=False,
        autosize=True,
    )

    # Add horizontal grid lines
    y_tick_spacing = 100 if total_height < 800 else 200
    y_grid = np.arange(0, total_height, y_tick_spacing)
    for y in y_grid:
        fig.add_shape(
            type="line",
            x0=0,
            y0=y,
            x1=-x_max,  # Extend grid line to the left
            y1=y,
            line=dict(color="lightgrey", width=0.5),
            layer="below",
        )

    # Adjust axes labels and ticks
    fig.update_xaxes(tickvals=[])

    fig.update_yaxes(
        zeroline=False,
        tickvals=[],
        showgrid=False,
    )

    # Vertical line at x=0 (y-axis)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=0,
        y1=y_max,
        line=dict(width=1, color=COLORS["lines"]),
        layer="below",
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
            y0=weaklayer_thickness,
            x1=x,
            y1=y_max,
            line=dict(color="lightgrey", width=0.5),
            layer="below",
        )

    # Horizontal line at table header
    fig.add_shape(
        type="line",
        x0=0,
        y0=column_header_y,
        x1=x_pos["col3_end"],
        y1=column_header_y,
        line=dict(color="lightgrey", width=0.5),
        layer="below",
    )

    # Annotations for table headers
    header_y_position = (y_max + column_header_y) / 2
    fig.add_annotation(
        x=(0 + x_pos["col1_start"]) / 2,
        y=header_y_position,
        text="H",  # "H<br>cm",  # "H (cm)",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=first_column_mid,
        y=header_y_position,
        text="D",  # 'D<br>kg/m³',  # "Density (kg/m³)",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=second_column_mid,
        y=header_y_position,
        text="F",  # "GF",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=third_column_mid,
        y=header_y_position,
        text="R",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )

    fig.add_annotation(
        x=-x_max,
        y=-substratum_thickness - 2,
        text="H – Height (cm)           D – Density (kg/m³)           F – Grain Form           R – Hand Hardness",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
    )

    # Adjust the plot margins (optional)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=40))

    return fig


def snow_profile_with_data(
    weaklayer: WeakLayer, layers: list[Layer], dataframe: pd.DataFrame
):
    fig = go.Figure()

    x_max_sserr = max(dataframe["sserr_result"])
    x_max_td = max(dataframe["touchdown_distance"])
    x_max_impact = max(dataframe["impact_criterion"])
    x_max_coupled = max(dataframe["coupled_criterion"])

    # Define colors for each axis
    AXIS_COLORS = {
        "sserr": "blue",
        "touchdown": "red",
        "impact": "green",
        "coupled": "orange",
    }

    fig.add_trace(
        go.Scatter(
            x=dataframe["sserr_result"] / 30,
            y=dataframe["wl_depth"],
            mode="lines+markers",
            name="SSERR",
            line=dict(color=AXIS_COLORS["sserr"], width=3),
            marker=dict(size=6, color=AXIS_COLORS["sserr"]),
            xaxis="x1",
        )
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=dataframe["touchdown_distance"],
    #         y=dataframe["wl_depth"],
    #         mode="lines+markers",
    #         name="Touchdown Distance",
    #         line=dict(color=AXIS_COLORS["touchdown"], width=3),
    #         marker=dict(size=6, color=AXIS_COLORS["touchdown"]),
    #         xaxis="x2",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=dataframe["impact_criterion"],
    #         y=dataframe["wl_depth"],
    #         mode="lines+markers",
    #         name="Impact Criterion",
    #         line=dict(color=AXIS_COLORS["impact"], width=3),
    #         marker=dict(size=6, color=AXIS_COLORS["impact"]),
    #         xaxis="x3",
    #     )
    # )
    fig.add_trace(
        go.Scatter(
            x=100 / dataframe["coupled_criterion"],
            y=dataframe["wl_depth"],
            mode="lines+markers",
            name="Coupled Criterion",
            line=dict(color=AXIS_COLORS["coupled"], width=3),
            marker=dict(size=6, color=AXIS_COLORS["coupled"]),
            xaxis="x3",
        )
    )

    # Configure multiple overlaying x-axes with enhanced colors and ticks
    fig.update_layout(
        # Main y-axis
        yaxis=dict(
            title="",  # Remove built-in title, we'll use annotation
            autorange="reversed",
            domain=[0.2, 1.0],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=2,
            tickmode="linear",
            tick0=0,
            dtick=50,  # Tick every 50 units
            tickcolor="black",
            tickwidth=2,
            ticklen=5,
        ),
        # First x-axis (SSERR) - primary axis
        xaxis=dict(
            title="",  # Remove built-in title, we'll use annotation
            range=[0, 5.0],
            side="bottom",
            autorange="reversed",
            showgrid=True,
            gridcolor="lightblue",
            gridwidth=1,
            tickmode="linear",
            tick0=0,
            dtick=max(x_max_sserr * 0.2, 1),  # 5 ticks across the range
            tickcolor=AXIS_COLORS["sserr"],
            tickwidth=2,
            ticklen=8,
            tickfont=dict(color=AXIS_COLORS["sserr"], size=10),
            linecolor=AXIS_COLORS["sserr"],
            linewidth=2,
        ),
        # # Second x-axis (Touchdown Distance)
        # xaxis2=dict(
        #     title="",  # Remove built-in title, we'll use annotation
        #     range=[0, x_max_td * 1.05],
        #     anchor="free",
        #     overlaying="x",
        #     side="bottom",
        #     position=0.15,
        #     autorange="reversed",
        #     showgrid=False,  # Avoid grid overlap
        #     tickmode="linear",
        #     tick0=0,
        #     dtick=max(x_max_td * 0.2, 1),  # 5 ticks across the range
        #     tickcolor=AXIS_COLORS["touchdown"],
        #     tickwidth=2,
        #     ticklen=8,
        #     tickfont=dict(color=AXIS_COLORS["touchdown"], size=10),
        #     linecolor=AXIS_COLORS["touchdown"],
        #     linewidth=2,
        # ),
        # Third x-axis (Impact Criterion)
        xaxis3=dict(
            title="",  # Remove built-in title, we'll use annotation
            range=[0.0, max(100 / dataframe["coupled_criterion"]) * 1.05],
            anchor="free",
            overlaying="x",
            side="bottom",
            position=0.1,
            zeroline=True,
            zerolinecolor=AXIS_COLORS["impact"],
            zerolinewidth=2,
            showgrid=False,  # Avoid grid overlap
            tickmode="linear",
            # autorange="reversed",
            tick0=0,
            dtick=max(x_max_impact * 0.2, 1),  # 5 ticks across the range
            tickcolor=AXIS_COLORS["impact"],
            tickwidth=2,
            ticklen=8,
            tickfont=dict(color=AXIS_COLORS["impact"], size=10),
            linecolor=AXIS_COLORS["impact"],
            linewidth=2,
        ),
        # # Fourth x-axis (Coupled Criterion)
        # xaxis4=dict(
        #     title="",  # Remove built-in title, we'll use annotation
        #     range=[-0.5, x_max_coupled * 1.05],
        #     anchor="free",
        #     overlaying="x",
        #     side="bottom",
        #     position=0.05,
        #     zeroline=True,
        #     zerolinecolor=AXIS_COLORS["coupled"],
        #     zerolinewidth=2,
        #     showgrid=False,  # Avoid grid overlap
        #     tickmode="linear",
        #     autorange="reversed",
        #     tick0=0,
        #     dtick=max(x_max_coupled * 0.2, 1),  # 5 ticks across the range
        #     tickcolor=AXIS_COLORS["coupled"],
        #     tickwidth=2,
        #     ticklen=8,
        #     tickfont=dict(color=AXIS_COLORS["coupled"], size=10),
        #     linecolor=AXIS_COLORS["coupled"],
        #     linewidth=2,
        # ),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        ),
        width=900,
        height=600,
        title=dict(
            text="Snow Profile Analysis - Multiple Criteria",
            font=dict(size=16, color="black"),
            x=0.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add custom annotations for axis titles positioned above the axis lines
    fig.add_annotation(
        text="Weak Layer Depth (cm)",
        x=-0.05,  # Position to the left of the plot
        y=0.6,  # Middle of the y-axis domain [0.2, 1.0]
        xref="paper",
        yref="paper",
        textangle=-90,  # Rotate 90 degrees counterclockwise
        font=dict(size=14, color="black"),
        showarrow=False,
        xanchor="center",
        yanchor="middle",
    )

    # X-axis title annotations positioned above their respective axes
    fig.add_annotation(
        text="SSERR (J/m²)",
        x=0.5,  # Center of the plot
        y=0.2,  # Just above the bottom axis
        xref="paper",
        yref="paper",
        font=dict(size=12, color=AXIS_COLORS["sserr"]),
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
    )

    # fig.add_annotation(
    #     text="Touchdown Distance (mm)",
    #     x=0.5,  # Center of the plot
    #     y=0.15,  # Above the position=0.15 axis (0.15 + 0.03)
    #     xref="paper",
    #     yref="paper",
    #     font=dict(size=12, color=AXIS_COLORS["touchdown"]),
    #     showarrow=False,
    #     xanchor="center",
    #     yanchor="bottom",
    # )

    fig.add_annotation(
        text="Critical Weight (kg)",
        x=0.5,  # Center of the plot
        y=0.1,  # Above the position=0.1 axis (0.1 + 0.03)
        xref="paper",
        yref="paper",
        font=dict(size=12, color=AXIS_COLORS["impact"]),
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
    )

    # fig.add_annotation(
    #     text="Critical Weight (kg)",
    #     x=0.5,  # Center of the plot
    #     y=0.05,  # Above the position=0.05 axis (0.05 + 0.03)
    #     xref="paper",
    #     yref="paper",
    #     font=dict(size=12, color=AXIS_COLORS["coupled"]),
    #     showarrow=False,
    #     xanchor="center",
    #     yanchor="bottom",
    # )

    return fig
