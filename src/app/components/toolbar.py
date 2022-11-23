"""Module describing the topic switcher component of the application."""
from dash_extensions.enrich import dcc, html

from app.components import topic_switcher, save_load
from app.utils.callback import init_callbacks

callbacks, def_callback = init_callbacks()
callbacks.extend(topic_switcher.callbacks)
callbacks.extend(save_load.callbacks)

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

toolbar_class = """
    flex transition-all ease-in delay-75
    fixed flex-row bottom-0 left-0 p-10
    w-full justify-between space-x-10
"""

layout = html.Div(
    id="topic_toolbar",
    className=toolbar_class,
    children=[save_load.layout, topic_switcher.layout, relevance_slider],
)
topic_toolbar = layout
