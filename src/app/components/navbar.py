# type: ignore
from dash import dcc, html

navbar = html.Div(
    className="""
    fixed flex flex-none flex-col justify-around content-middle
    top-1/2 -right-0 w-20 h-1/6 z-10 mr-5
    text-center
    bg-white shadow rounded-full
    -translate-y-1/2
    """,
    children=[html.Button("Documents"), html.Button("Topics")],
)
