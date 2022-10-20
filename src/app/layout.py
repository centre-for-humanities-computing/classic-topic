# type: ignore
"""Module describing the layout of the app"""

import dash
from dash import dcc, html

from app.components.topic_switcher import topic_switcher
from app.components.sidebar import sidebar
from app.components.navbar import navbar
from app.components.save_load import save_load
from app.components.genre_weight_popup import genre_weight_popup

view_class = "flex-row items-stretch flex-1 mr-16 z-0"

layout = html.Div(
    className="flex flex-row w-full h-full fixed",
    children=[
        dcc.Store(id="fit_store", storage_type="session"),
        dcc.Store(id="topic_names", storage_type="session"),
        dcc.Store(id="current_topic", data={"current_topic": 0}),
        dcc.Store(id="genre_weights", storage_type="session"),
        dcc.Store(id="genre_names", data=[], storage_type="session"),
        dcc.Interval(id="fetch_genres", disabled=False),
        dcc.Store(
            id="current_view",
            storage_type="session",
            data={"current_view": "topic"},
        ),
        html.Div(
            id="topic_view",
            className=view_class + " flex mb-16",
            children=[
                dcc.Graph(
                    id="all_topics_plot", className="flex-1 basis-1/3 mt-10"
                ),
                dcc.Graph(
                    id="current_topic_plot", className="flex-1 basis-2/3"
                ),
            ],
        ),
        html.Div(
            id="document_view",
            className=view_class + " hidden",
            children=dcc.Graph(id="all_documents_plot", className="flex-1"),
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
        save_load,
        # genre_weight_popup
    ],
)


def add_layout(app: dash.Dash) -> None:
    """Adds layout to Dash app"""
    app.layout = layout
