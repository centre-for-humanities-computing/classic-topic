# type: ignore
import dash
from dash import dcc, html

from app.components.topic_switcher import topic_switcher
from app.components.sidebar import sidebar

layout = html.Div(
    className="flex flex-row w-full h-full fixed",
    children=[
        dcc.Store(id="fit_store", storage_type="local"),
        dcc.Store(id="topic_names", storage_type="local"),
        dcc.Store(id="current_topic", storage_type="local"),
        dcc.Graph(id="main_plot", className="flex-1 mr-16 z-0"),
        topic_switcher,
        sidebar,
    ],
)


def add_layout(app: dash.Dash) -> None:
    """Adds layout to Dash app"""
    app.layout = layout
