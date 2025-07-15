# Third-party imports
import plotly.graph_objects as go


def plot_traffic_light(bar_position, theme):
    # Define box labels and colors
    labels = ["good", "fair", "poor", "very poor"]
    box_colors = ["#C1E67E", "#FFDA62", "#F7AB50", "#C70039"]
    bg_color = theme["backgroundColor"]
    bar_color = theme["textColor"]
    if theme["base"] == "dark":
        gray_color = "darkgray"
    else:
        gray_color = "lightgray"

    # Define box positions with a small gap between them
    gap = 0.01
    box_width = (1 - 3 * gap) / 4
    positions = [i * (box_width + gap) for i in range(len(labels))]

    # Create box shapes with correct coloring
    shapes = []
    for i, pos in enumerate(positions):
        if (
            (i == 0 and bar_position <= 0.25)
            or (i == 1 and 0.25 < bar_position <= 0.5)
            or (i == 2 and 0.5 < bar_position <= 0.75)
            or (i == 3 and 0.75 < bar_position <= 1)
        ):
            fill_color = box_colors[i]
        else:
            fill_color = gray_color

        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": pos,
                "x1": pos + box_width,
                "y0": 0.4,
                "y1": 0.9,
                "fillcolor": fill_color,
                "opacity": 1,
                "line": {"width": 0},  # No outline
                "layer": "below",
            }
        )

    # Create the vertical bar extending above and below the boxes
    shapes.append(
        {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": bar_position,
            "x1": bar_position,
            "y0": 0.3,
            "y1": 1,
            "line": {"color": bg_color, "width": 7},
        }
    )
    shapes.append(
        {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": bar_position,
            "x1": bar_position,
            "y0": 0.3,
            "y1": 1,
            "line": {"color": bar_color, "width": 2},
        }
    )

    # Create the figure
    fig = go.Figure()

    # Add shapes to the figure
    fig.update_layout(
        shapes=shapes,
        xaxis={
            "range": [0, 1],
            "showgrid": False,
            "zeroline": False,
            "visible": False,
        },
        yaxis={
            "range": [0, 1],
            "showgrid": False,
            "zeroline": False,
            "visible": False,
        },
        height=50,
        width=800,
        margin=dict(t=0, b=0, l=0, r=0),
    )

    # Add labels as annotations below the boxes
    for i, pos in enumerate(positions):
        fig.add_annotation(
            x=pos + box_width / 2,
            y=0.15,
            text=labels[i],
            showarrow=False,
            font=dict(size=12),
        )

    return fig
