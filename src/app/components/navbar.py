# type: ignore
from dash import dcc, html

button_class = """
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
            className=button_class + " text-sky-700",
        ),
        html.Button(
            html.I(className="fa-solid fa-book"),
            title="Documents",
            className=button_class + " text-gray-500",
        ),
    ],
)
