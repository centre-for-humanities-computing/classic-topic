from typing import List

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import dcc, html
from dash_extensions.enrich import Input, Output, State

from app.utils.callback import init_callbacks

callbacks, def_callback = init_callbacks()

topic_switcher_class = """
    flex h-16 flex-row bg-white bg-opacity-100 shadow z-10 rounded-full
    justify-between px-6
    basis-1/2 shrink
"""

layout = html.Div(
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


@def_callback(
    Output("next_topic", "children"),
    Output("next_topic", "disabled"),
    Output("prev_topic", "children"),
    Output("prev_topic", "disabled"),
    Output("topic_name", "value"),
    State("topic_names", "data"),
    Input("current_topic", "data"),
    prevent_initial_call=True,
)
def update_topic_switcher(topic_names: List[str], current_topic: int):
    """Updates the topic switcher component when the current topic changes."""
    if not topic_names:
        raise PreventUpdate
    n_topics = len(topic_names)
    current = topic_names[current_topic]
    prev_disabled = current_topic == 0
    next_disabled = current_topic == n_topics - 1
    prev_topic = "" if prev_disabled else "<- " + topic_names[current_topic - 1]
    next_topic = "" if next_disabled else topic_names[current_topic + 1] + " ->"
    return next_topic, next_disabled, prev_topic, prev_disabled, current
