# type: ignore
"""Module describing the navigation bar component of the app."""
from dash_extensions.enrich import dcc, html

navbar_button_class = """
    p-3 text-xl transition-all ease-in
    hover:text-sky-600
"""

navbar = html.Div(
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
