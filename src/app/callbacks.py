from typing import Any, Callable, Dict, Hashable, List, Tuple, TypeVar

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.components.sidebar import sidebar_body_class
from app.components.topic_switcher import topic_switcher_class
from app.utils.modelling import (
    calculate_importance,
    calculate_top_words,
    fit_pipeline,
    load_corpus,
)
from app.utils.plots import genre_plot, join_plots, word_plot

callbacks = []

T = TypeVar("T")

Wrapped = Dict[str, T]

DictFrame = Dict[Hashable, Any]


def cb(*args, **kwargs) -> Callable:
    """Decorator to add function to the global callback list"""

    def _cb(func: Callable):
        callbacks.append({"function": func, "args": args, "kwargs": kwargs})
        return func

    return _cb


def add_callbacks(app: dash.Dash) -> None:
    """Adds the list of callbacks to a Dash app."""
    for callback in callbacks:
        dash.callback(app=app, *callback["args"], **callback["kwargs"])(
            callback["function"]
        )


@cb(
    Output("fit_store", "data"),
    Output("loading", "children"),
    Input("fit_pipeline", "n_clicks"),
    State("select_vectorizer", "value"),
    State("min_df", "value"),
    State("max_df", "value"),
    State("select_model", "value"),
    State("n_topics", "value"),
    prevent_initial_call=True,
)
def update_fit(
    n_clicks: int,
    vectorizer_name: str,
    min_df: int,
    max_df: float,
    model_name: str,
    n_topics: int,
) -> Tuple[Dict, List]:
    if not n_clicks:
        raise PreventUpdate
    corpus = load_corpus()
    pipeline = fit_pipeline(
        corpus=corpus,
        vectorizer_name=vectorizer_name,
        min_df=min_df,
        max_df=max_df,
        model_name=model_name,
        n_topics=n_topics,
    )
    importance = calculate_importance(corpus, pipeline)
    top_words = calculate_top_words(pipeline, top_n=30)
    return (
        {
            "importance": importance.to_dict(),
            "top_words": top_words.to_dict(),
            "n_topics": n_topics,
        },
        [],
    )


@cb(
    Output("next_topic", "children"),
    Output("next_topic", "disabled"),
    Output("prev_topic", "children"),
    Output("prev_topic", "disabled"),
    Output("topic_name", "value"),
    State("topic_names", "data"),
    Input("current_topic", "data"),
    prevent_initial_call=True,
)
def update_topic_switcher(
    topic_names_data: Wrapped[List[str]], current_topic_data: Wrapped[int]
):
    if topic_names_data is None or current_topic_data is None:
        raise PreventUpdate
    topic_names = topic_names_data["topic_names"]
    current_topic = current_topic_data["current_topic"]
    n_topics = len(topic_names)
    current = topic_names[current_topic]
    prev_disabled = current_topic == 0
    next_disabled = current_topic == n_topics - 1
    prev_topic = (
        "" if prev_disabled else "<- " + topic_names[current_topic - 1]
    )
    next_topic = (
        "" if next_disabled else topic_names[current_topic + 1] + " ->"
    )
    return next_topic, next_disabled, prev_topic, prev_disabled, current


@cb(
    Output("topic_names", "data"),
    State("topic_names", "data"),
    State("current_topic", "data"),
    Input("topic_name", "value"),
    Input("fit_store", "data"),
    prevent_initial_call=True,
)
def update_topic_names(
    topic_names_data: Wrapped[List[str]],
    current_topic_data: Wrapped[int],
    topic_name: str,
    fit_store: Dict,
) -> Wrapped[List[str]]:
    if ctx.triggered_id == "fit_store":
        return {
            "topic_names": [f"Topic {i}" for i in range(fit_store["n_topics"])]
        }
    if topic_names_data is None or current_topic_data is None:
        raise PreventUpdate()
    topic_names = topic_names_data["topic_names"]
    current_topic = current_topic_data["current_topic"]
    if not topic_names:
        raise PreventUpdate()
    new_names = topic_names.copy()
    new_names[current_topic] = topic_name
    return {"topic_names": new_names}


@cb(
    Output("sidebar_body", "className"),
    Output("topic_switcher", "className"),
    Input("sidebar_collapser", "n_clicks"),
    prevent_initial_call=True,
)
def open_close_sidebar(n_clicks: int) -> Tuple[str, str]:
    is_open = (n_clicks % 2) == 1
    if is_open:
        return (
            sidebar_body_class + " translate-x-full",
            topic_switcher_class + " -translate-x-1/2",
        )
    else:
        return (
            sidebar_body_class + " translate-x-0",
            topic_switcher_class + " -translate-x-2/3",
        )


@cb(
    Output("current_topic", "data"),
    State("current_topic", "data"),
    Input("fit_store", "data"),
    Input("next_topic", "n_clicks"),
    Input("prev_topic", "n_clicks"),
    prevent_initial_call=True,
)
def update_current_topic(
    current_topic_data: Wrapped[int],
    fit_store: Dict,
    next_clicks: int,
    prev_clicks: int,
) -> Wrapped[int]:
    if "fit_store" == ctx.triggered_id:
        return {"current_topic": 0}
    if not next_clicks and not prev_clicks:
        raise PreventUpdate()
    if current_topic_data is None or "current_topic" not in current_topic_data:
        raise PreventUpdate()
    current_topic = current_topic_data["current_topic"]
    if ctx.triggered_id == "next_topic":
        return {"current_topic": current_topic + 1}
    elif ctx.triggered_id == "prev_topic":
        return {"current_topic": current_topic - 1}
    else:
        raise PreventUpdate()


@cb(
    Output("main_plot", "figure"),
    Input("current_topic", "data"),
    Input("fit_store", "data"),
    prevent_initial_call=True,
)
def update_plot(
    current_topic_data: Wrapped[int], fit_store: Dict
) -> go.Figure:
    if current_topic_data is None or fit_store is None:
        raise PreventUpdate()
    current_topic = current_topic_data["current_topic"]
    importance = pd.DataFrame(fit_store["importance"])
    top_words = pd.DataFrame(fit_store["top_words"])
    return join_plots(
        genre_plot(current_topic, importance),
        word_plot(current_topic, top_words),
    )
