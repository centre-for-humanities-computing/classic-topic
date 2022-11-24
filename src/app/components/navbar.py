"""Module describing the navigation bar component of the app."""
from typing import Tuple

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import html
from dash_extensions.enrich import Input, Output
from dash import ctx

from app.views.topic_view import view_class
from app.utils.callback import init_callbacks

callbacks, def_callback = init_callbacks()

navbar_button_class = """
    p-3 text-xl transition-all ease-in
    hover:text-sky-600
"""

layout = html.Div(
    className="""
    fixed flex flex-none flex-col justify-around content-middle
    top-1/2 -right-0 z-10 mr-5 p-3
    text-center
    bg-white shadow rounded-full
    -translate-y-1/2
    """,
    children=[
        html.Button(
            html.I(className="fa-solid fa-hashtag"),
            title="Topics",
            id="topic_view_button",
            className=navbar_button_class + " text-sky-700",
            n_clicks=0,
        ),
        html.Button(
            html.I(className="fa-solid fa-book"),
            title="Documents",
            id="document_view_button",
            className=navbar_button_class + " text-gray-500",
            n_clicks=0,
        ),
    ],
)


@def_callback(
    Output("current_view", "data"),
    Input("topic_view_button", "n_clicks"),
    Input("document_view_button", "n_clicks"),
    prevent_initial_call=True,
)
def update_current_view(topic_clicks: int, document_clicks: int) -> str:
    """Updates the current view value in store when a view is selected
    on the navbar."""
    if not topic_clicks and not document_clicks:
        raise PreventUpdate()
    if ctx.triggered_id == "topic_view_button":
        return "topic"
    if ctx.triggered_id == "document_view_button":
        return "document"
    raise PreventUpdate()


@def_callback(
    Output("topic_view_button", "className"),
    Output("document_view_button", "className"),
    Output("topic_view", "className"),
    Output("document_view", "className"),
    Input("current_view", "data"),
    prevent_initial_call=True,
)
def switch_views(current_view: str) -> Tuple[str, str, str, str]:
    """Switches views when the current view value in the store is changed."""
    if current_view == "topic":
        return (
            navbar_button_class + " text-sky-700",
            navbar_button_class + " text-gray-500",
            view_class + " flex mb-16",
            view_class + " hidden",
        )
    if current_view == "document":
        return (
            navbar_button_class + " text-gray-500",
            navbar_button_class + " text-sky-700",
            view_class + " hidden mb-16",
            view_class + " flex",
        )
    raise PreventUpdate()
