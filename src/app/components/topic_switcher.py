# type: ignore
"""Module describing the topic switcher component of the application."""
from dash import html, dcc

topic_switcher_class = """
    fixed min-w-fit w-1/2 h-16 flex-row bg-white
    bg-opacity-100 shadow
    bottom-10 left-1/2
    justify-start z-10
    content-center rounded-full
    transition-all ease-in delay-75
"""

topic_switcher = html.Div(
    id="topic_switcher",
    className=topic_switcher_class + " -translate-x-1/2 flex",
    children=[
        html.Button(
            "<- Previous topic",
            id="prev_topic",
            n_clicks=0,
            className="flex-1 text-sky-700 hover:text-sky-800",
        ),
        html.P(
            "Rename current topic: ",
            className="""
            self-center flex-0 italic text-gray-500 text-sm
            """,
        ),
        dcc.Input(
            id="topic_name",
            type="text",
            value="Topic 1",
            debounce=True,
            className="""
            flex-0 w-min-fit mx-5 text-center text-lg px-5
            text-gray-600 border-b-2 border-gray-200 border-dashed
            focus:text-gray-900 focus:border-sky-200 focus:border-solid
            """,
        ),
        html.Button(
            "Next topic ->",
            id="next_topic",
            n_clicks=0,
            className="flex-1 text-sky-700 hover:text-sky-800",
        ),
    ],
)
