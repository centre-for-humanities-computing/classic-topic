# type: ignore
"""Module describing the topic switcher component of the application."""
from dash_extensions.enrich import dcc, html

topic_switcher_class = """
    flex h-16 flex-row bg-white bg-opacity-100 shadow z-10 rounded-full
    justify-between px-6
    basis-1/2 shrink
"""

topic_switcher = html.Div(
    id="topic_switcher",
    className=topic_switcher_class,
    children=[
        html.Button(
            "<- Previous topic",
            id="prev_topic",
            n_clicks=0,
            className="flex-1 mr-3 text-sky-700 hover:text-sky-800",
        ),
        html.P(
            "Rename current topic: ",
            className="""
            self-center italic text-gray-500 text-sm 
            """,
        ),
        dcc.Input(
            id="topic_name",
            type="text",
            value="Topic 1",
            debounce=True,
            className="""
            mx-5 text-center text-lg px-3 flex-1
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

relevance_slider = html.Div(
    className="""
        flex justify-between items-center
        h-16 bg-white shadow shrink basis-1/4
        rounded-full
        px-6 py-6 flex-nowrap
    """,
    children=[
        html.Div("λ :", className="text-xl text-gray-500"),
        dcc.Slider(
            id="lambda_slider",
            value=1.0,
            min=0.0,
            max=1.0,
            className="flex-1 mt-5",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ],
)

button_class = """
    text-xl transition-all ease-in 
    justify-center content-center items-center
    text-gray-500 hover:text-sky-600
    flex flex-1
"""

save_load = html.Div(
    className="""
        flex flex-none flex-row justify-center content-middle
        h-16 bg-white shadow rounded-full w-32
        rounded-full ml-5
    """,
    children=[
        html.Button(
            html.I(className="fa-solid fa-file-arrow-down"),
            id="download_button",
            title="Download data",
            className=button_class,
        ),
        dcc.Download(id="download"),
        html.Div(
            className=button_class,
            children=dcc.Upload(
                id="upload",
                children=html.Button(
                    html.I(className="fa-solid fa-file-arrow-up"),
                    id="upload_button",
                    title="Upload data",
                ),
            ),
        ),
    ],
)
toolbar_class = """
    flex transition-all ease-in delay-75
    fixed flex-row bottom-0 left-0 p-10
    w-full justify-between space-x-10
"""

topic_toolbar = html.Div(
    id="topic_toolbar",
    className=toolbar_class,
    children=[save_load, topic_switcher, relevance_slider],
)
