from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from dash import ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State, dcc, html

from app.utils.callback import init_callbacks
from app.utils.modelling import calculate_top_words
from app.utils.plots import all_topics_plot, topic_plot

callbacks, def_callback = init_callbacks()

view_class = "flex-row items-stretch flex-1 mr-16 z-0"

layout = html.Div(
    id="topic_view",
    className=view_class + " flex mb-16",
    children=[
        dcc.Store(id="current_topic", data=0),
        dcc.Graph(
            id="all_topics_plot",
            className="flex-1 basis-1/3 mt-10",
            responsive=True,
            config=dict(scrollZoom=True),
            animate=True,
        ),
        dcc.Graph(
            id="current_topic_plot",
            className="flex-1 basis-2/3",
            responsive=True,
        ),
    ],
)


@def_callback(
    Output("current_topic_plot", "figure"),
    Input("current_topic", "data"),
    Input("fit_store", "data"),
    Input("lambda_slider", "value"),
    prevent_initial_call=True,
)
def update_current_topic_plot(
    current_topic: int, fit_store: Dict, alpha: float
) -> go.Figure:
    """Updates the plots about the current topic in the topic view
    when the current topic is changed or when a new model is fitted.
    """
    if current_topic is None or fit_store is None or alpha is None:
        raise PreventUpdate()
    top_words = calculate_top_words(
        current_topic=current_topic, top_n=30, alpha=alpha, **fit_store
    )
    genre_importance = pd.DataFrame(fit_store["genre_importance"])
    genre_importance = genre_importance[
        genre_importance.topic == current_topic
    ]
    return topic_plot(top_words=top_words, genre_importance=genre_importance)


@def_callback(
    Output("current_topic", "data"),
    State("current_topic", "data"),
    Input("fit_store", "data"),
    Input("next_topic", "n_clicks"),
    Input("prev_topic", "n_clicks"),
    Input("all_topics_plot", "clickData"),
    prevent_initial_call=True,
)
def update_current_topic(
    current_topic: int,
    fit_store: Dict,
    next_clicks: int,
    prev_clicks: int,
    plot_click_data: Dict,
) -> int:
    """Updates current topic in the store when one is selected."""
    if "fit_store" == ctx.triggered_id:
        return 0
    if "all_topics_plot" == ctx.triggered_id:
        if plot_click_data is None:
            raise PreventUpdate()
        # In theory multiple points could be selected with
        # multiple customdata elements, so we unpack the first element.
        point, *_ = plot_click_data["points"]
        topic_id, *_ = point["customdata"]
        return topic_id
    if not next_clicks and not prev_clicks:
        raise PreventUpdate()
    if ctx.triggered_id == "next_topic":
        return current_topic + 1
    elif ctx.triggered_id == "prev_topic":
        return current_topic - 1
    else:
        raise PreventUpdate()


@def_callback(
    Output("all_topics_plot", "figure"),
    Input("fit_store", "data"),
    Input("topic_names", "data"),
    Input("current_topic", "data"),
    prevent_initial_call=True,
)
def update_all_topics_plot(
    fit_data: Dict,
    topic_names: List[str],
    current_topic: int,
) -> go.Figure:
    """Updates the topic overview plot when the fit, the topic names or the
    current topic change."""
    if not topic_names or fit_data is None:
        # If there's missing data, prevent update.
        raise PreventUpdate()
    x, y = fit_data["topic_pos"]
    size = fit_data["topic_frequency"]
    topic_id = fit_data["topic_id"]
    topic_data = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "size": size,
            "topic_id": topic_id,
        }
    )
    # Mapping topic names over to topic ids with a Series
    # since Series also function as a mapping, you can use them in the .map()
    # method
    names = pd.Series(topic_names)
    topic_data = topic_data.assign(topic_name=topic_data.topic_id.map(names))
    print(f"{topic_data=}")
    try:
        fig = all_topics_plot(topic_data, current_topic)
    except ValueError as e:
        print(f"Failed to plot topics, error: {e}")
        raise PreventUpdate
    return fig
