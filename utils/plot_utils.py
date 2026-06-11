import plotly.graph_objects as go
import plotly.express as px

# Brand color scheme
PRIMARY_BLUE = "#378ADD"
ACCENT_TEAL = "#1D9E75"
WARNING_AMBER = "#EF9F27"
DANGER_RED = "#E24B4A"
COLOR_SEQUENCE = [PRIMARY_BLUE, ACCENT_TEAL, WARNING_AMBER, DANGER_RED, "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

def base_layout():
    """
    Returns standard base layout configuration dictionary.
    """
    return {
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": dict(family="Inter, sans-serif", size=12),
        "gridcolor": "rgba(128,128,128,0.15)",
        "colors": {
            "blue": "#378ADD",
            "teal": "#1D9E75",
            "amber": "#EF9F27",
            "red": "#E24B4A"
        }
    }

def apply_layout_theme(fig, title=None):
    """
    Applies standard, polished theme configurations to a Plotly figure.
    """
    layout_cfg = base_layout()
    fig.update_layout(
        title=dict(
            text=title if title else "",
            font=dict(family=layout_cfg["font"]["family"], size=18, color="#ffffff"),
            x=0.0,
            y=0.95
        ),
        paper_bgcolor=layout_cfg["paper_bgcolor"],
        plot_bgcolor=layout_cfg["plot_bgcolor"],
        font=layout_cfg["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#d4d4d8", family=layout_cfg["font"]["family"])
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(
            bgcolor="#1e1e2e",
            font_size=13,
            font_family=layout_cfg["font"]["family"],
            font_color="#ffffff"
        )
    )
    
    # Configure axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor=layout_cfg["gridcolor"],
        linecolor="rgba(128,128,128,0.15)",
        tickfont=dict(color="#a1a1aa", family=layout_cfg["font"]["family"]),
        title_font=dict(color="#d4d4d8", family=layout_cfg["font"]["family"])
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=layout_cfg["gridcolor"],
        linecolor="rgba(128,128,128,0.15)",
        tickfont=dict(color="#a1a1aa", family=layout_cfg["font"]["family"]),
        title_font=dict(color="#d4d4d8", family=layout_cfg["font"]["family"])
    )
    return fig

def create_line_chart(df, x_col, y_col, color_col=None, title=None, labels=None):
    """Creates a line chart with consistent branding."""
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        labels=labels,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    # Style line traces
    fig.update_traces(line=dict(width=2.5))
    return apply_layout_theme(fig, title)

def create_bar_chart(df, x_col, y_col, color_col=None, title=None, barmode="group", labels=None):
    """Creates a bar chart with consistent branding."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode=barmode,
        labels=labels,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    return apply_layout_theme(fig, title)

def create_donut_chart(df, names_col, values_col, title=None):
    """Creates a donut chart with consistent branding."""
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        hole=0.4,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return apply_layout_theme(fig, title)

def create_heatmap(corr_df, title=None):
    """Creates a correlation matrix heatmap with the blue/teal sequential palette."""
    # Create custom colorscale from teal to blue
    colorscale = [
        [0.0, "#1D9E75"],
        [0.5, "#ffffff"],
        [1.0, "#378ADD"]
    ]
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=colorscale
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=60, r=40, t=60, b=40)
    )
    return apply_layout_theme(fig, title)

def create_histogram(df, x_col, title=None, labels=None):
    """Creates a distribution histogram."""
    fig = px.histogram(
        df,
        x=x_col,
        labels=labels,
        color_discrete_sequence=[PRIMARY_BLUE]
    )
    fig.update_layout(bargap=0.05)
    return apply_layout_theme(fig, title)
