from typing import Callable, Dict, Tuple

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.utils.modelling import (
    calculate_importance,
    calculate_top_words,
    fit_pipeline,
    load_corpus,
)

from app.components.sidebar import sidebar_body_class
from app.components.topic_switcher import topic_switcher_class

callbacks = []


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
    Output("topic_names", "data"),
    Output("current_topic", "data"),
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
) -> Tuple[Dict, Dict, Dict]:
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
        },
        {"topic_names": [f"Topic {i}" for i in range(n_topics)]},
        {"current_topic": 0},
    )


@cb(
    Output("next_topic", "children"),
    Output("next_topic", "disabled"),
    Output("prev_topic", "children"),
    Output("prev_topic", "disabled"),
    Output("topic_name", "value"),
    Input("topic_names", "data"),
    Input("current_topic", "data"),
    prevent_initial_call=True,
)
def update_topic_switcher(topic_names, current_topic):
    if topic_names is None or current_topic is None:
        raise PreventUpdate
    topic_names = topic_names["topic_names"]
    current_topic = current_topic["current_topic"]
    n_topics = len(topic_names)
    current = topic_names[current_topic]
    prev_disabled = current_topic == 0
    next_disabled = current_topic == n_topics - 1
    prev_topic = "" if prev_disabled else topic_names[current_topic - 1]
    next_topic = "" if next_disabled else topic_names[current_topic + 1]
    return next_topic, next_disabled, prev_topic, prev_disabled, current


@cb(
    Output("sidebar_body", "className"),
    Output("topic_switcher", "className"),
    Input("sidebar_collapser", "n_clicks"),
    prevent_initial_call=True,
)
def open_close_sidebar(n_clicks: int):
    is_open = (n_clicks % 2) == 0
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
