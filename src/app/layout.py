# type: ignore
import dash
from dash import dcc, html

import app.styles as styles

from app import components


layout = html.Div(
    style=styles.window,
    children=[
        dcc.Store(id="fit_store", storage_type="local"),
        dcc.Store(id="topic_names", storage_type="local"),
        dcc.Store(id="current_topic", storage_type="local"),
        html.Div(
            style={
                **styles.page_visible,
                "display": "flex",
                "flex-direction": "column",
            },
            children=[
                html.Div(
                    id="plot_container",
                    style={
                        "display": "flex",
                        "flex": "15 0",
                    },
                    children=dcc.Graph(
                        id="plot_space",
                        style={"height": "100%", "width": "100%"},
                    ),
                ),
                components.topic_switcher,
            ],
        ),
        html.Div(
            id="sidebar",
            style=styles.sidebar,
            children=[components.sidebar_collapser, components.sidebar_body],
        ),
    ],
)


def add_layout(app: dash.Dash) -> None:
    """Adds layout to Dash app"""
    app.layout = layout
