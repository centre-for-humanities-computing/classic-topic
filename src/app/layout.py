# type: ignore
import dash
from dash import dcc, html

from app.components.topic_switcher import topic_switcher
from app.components.sidebar import sidebar
from app.components.navbar import navbar

layout = html.Div(
    className="flex flex-row w-full h-full fixed",
    children=[
        dcc.Store(id="fit_store", storage_type="local"),
        dcc.Store(id="topic_names", storage_type="local"),
        dcc.Store(id="current_topic", storage_type="local"),
        html.Div(
            id="topic_view",
            className="flex flex-row items-stretch flex-1 mr-16 mb-16 z-0",
            children=[
                dcc.Graph(
                    id="all_topics_plot", className="flex-1 basis-1/3 mt-10"
                ),
                dcc.Graph(
                    id="current_topic_plot", className="flex-1 basis-2/3"
                ),
            ],
        ),
        dcc.Loading(
            type="circle",
            className="flex flex-1 mr-16 z-0",
            fullscreen=True,
            children=html.Div(id="loading"),
        ),
        topic_switcher,
        sidebar,
        navbar,
    ],
)


def add_layout(app: dash.Dash) -> None:
    """Adds layout to Dash app"""
    app.layout = layout
