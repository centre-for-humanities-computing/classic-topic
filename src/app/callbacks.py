import json
from typing import Any, Callable, Dict, Hashable, List, Tuple, TypeVar

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.components.sidebar import sidebar_body_class
from app.components.topic_switcher import topic_switcher_class
from app.components.navbar import navbar_button_class
from app.layout import view_class
from app.utils.modelling import (
    calculate_document_data,
    calculate_genre_importance,
    calculate_top_words,
    calculate_topic_data,
    fit_pipeline,
    load_corpus,
)
from app.utils.plots import all_topics_plot, documents_plot, topic_plot

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
    genre_importance = calculate_genre_importance(corpus, pipeline)
    top_words = calculate_top_words(pipeline, top_n=30)
    topic_data = calculate_topic_data(corpus, pipeline)
    document_data = calculate_document_data(corpus, pipeline)
    return (
        {
            "genre_importance": genre_importance.to_dict(),
            "top_words": top_words.to_dict(),
            "topic_data": topic_data.to_dict(),
            "document_data": document_data.to_dict(),
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
        if fit_store is None:
            raise PreventUpdate()
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
    Output("sidebar_collapser", "children"),
    Input("sidebar_collapser", "n_clicks"),
    Input("current_view", "data"),
    prevent_initial_call=True,
)
def open_close_sidebar(
    n_clicks: int, current_view_data: Wrapped[str]
) -> Tuple[str, str, str]:
    if n_clicks is None or current_view_data is None:
        raise PreventUpdate()
    view = current_view_data["current_view"]
    hide_switcher = " flex" if view == "topic" else " hidden"
    is_open = (n_clicks % 2) == 0
    if is_open:
        return (
            sidebar_body_class + " translate-x-full",
            topic_switcher_class + " -translate-x-1/2" + hide_switcher,
            "⚙️",
        )
    else:
        return (
            sidebar_body_class + " translate-x-0",
            topic_switcher_class + " -translate-x-2/3" + hide_switcher,
            "✕",
        )


@cb(
    Output("sidebar_collapser", "n_clicks"),
    State("sidebar_collapser", "n_clicks"),
    Input("fit_store", "data"),
    Input("fit_pipeline", "n_clicks"),
    prevent_initial_call=True,
)
def open_close_sidebar_fitting(
    current: int, fit_data: Dict, n_clicks: int
) -> int:
    if ((ctx.triggered_id == "fit_store") and (fit_data is None)) or (
        (ctx.triggered_id == "fit_pipeline") and n_clicks
    ):
        return current + 1
    else:
        raise PreventUpdate()


@cb(
    Output("current_view", "data"),
    Input("topic_view_button", "n_clicks"),
    Input("document_view_button", "n_clicks"),
    prevent_initial_call=True,
)
def update_current_view(
    topic_clicks: int, document_clicks: int
) -> Wrapped[str]:
    if not topic_clicks and not document_clicks:
        raise PreventUpdate()
    if ctx.triggered_id == "topic_view_button":
        return {"current_view": "topic"}
    if ctx.triggered_id == "document_view_button":
        return {"current_view": "document"}
    raise PreventUpdate()


@cb(
    Output("topic_view_button", "className"),
    Output("document_view_button", "className"),
    Output("topic_view", "className"),
    Output("document_view", "className"),
    Input("current_view", "data"),
    prevent_initial_call=True,
)
def switch_views(current_view_data: Wrapped[str]) -> Tuple[str, str, str, str]:
    if not current_view_data or "current_view" not in current_view_data:
        raise PreventUpdate()
    view = current_view_data["current_view"]
    if view == "topic":
        return (
            navbar_button_class + " text-sky-700",
            navbar_button_class + " text-gray-500",
            view_class + " flex mb-16",
            view_class + " hidden",
        )
    if view == "document":
        return (
            navbar_button_class + " text-gray-500",
            navbar_button_class + " text-sky-700",
            view_class + " hidden mb-16",
            view_class + " flex",
        )
    raise PreventUpdate()


@cb(
    Output("current_topic", "data"),
    State("current_topic", "data"),
    Input("fit_store", "data"),
    Input("next_topic", "n_clicks"),
    Input("prev_topic", "n_clicks"),
    Input("all_topics_plot", "clickData"),
    prevent_initial_call=True,
)
def update_current_topic(
    current_topic_data: Wrapped[int],
    fit_store: Dict,
    next_clicks: int,
    prev_clicks: int,
    plot_click_data: Dict,
) -> Wrapped[int]:
    if "fit_store" == ctx.triggered_id:
        return {"current_topic": 0}
    if current_topic_data is None or "current_topic" not in current_topic_data:
        raise PreventUpdate()
    current_topic = current_topic_data["current_topic"]
    if "all_topics_plot" == ctx.triggered_id:
        if plot_click_data is None:
            raise PreventUpdate()
        point, *_ = plot_click_data["points"]
        topic_id, *_ = point["customdata"]
        return {"current_topic": topic_id}
    if not next_clicks and not prev_clicks:
        raise PreventUpdate()
    if ctx.triggered_id == "next_topic":
        return {"current_topic": current_topic + 1}
    elif ctx.triggered_id == "prev_topic":
        return {"current_topic": current_topic - 1}
    else:
        raise PreventUpdate()


@cb(
    Output("current_topic_plot", "figure"),
    Input("current_topic", "data"),
    Input("fit_store", "data"),
    prevent_initial_call=True,
)
def update_current_topic_plot(
    current_topic_data: Wrapped[int], fit_store: Dict
) -> go.Figure:
    if current_topic_data is None or fit_store is None:
        raise PreventUpdate()
    current_topic = current_topic_data["current_topic"]
    genre_importance = pd.DataFrame(fit_store["genre_importance"])
    top_words = pd.DataFrame(fit_store["top_words"])
    return topic_plot(
        current_topic, genre_importance=genre_importance, top_words=top_words
    )


@cb(
    Output("all_topics_plot", "figure"),
    Input("fit_store", "data"),
    Input("topic_names", "data"),
    Input("current_topic", "data"),
    prevent_initial_call=True,
)
def update_all_topics_plot(
    fit_data: Dict,
    topic_names_data: Wrapped[List[str]],
    current_topic_data: Wrapped[int],
) -> go.Figure:
    if (
        fit_data is None
        or topic_names_data is None
        or current_topic_data is None
        or ("topic_data" not in fit_data)
        or ("topic_names" not in topic_names_data)
        or ("current_topic" not in current_topic_data)
    ):
        raise PreventUpdate()
    topic_data = pd.DataFrame(fit_data["topic_data"])
    current_topic = current_topic_data["current_topic"]
    names = pd.Series(topic_names_data["topic_names"])
    topic_data = topic_data.assign(topic_name=topic_data.topic_id.map(names))
    fig = all_topics_plot(topic_data, current_topic)
    return fig


@cb(
    Output("all_documents_plot", "figure"),
    Input("fit_store", "data"),
    Input("topic_names", "data"),
)
def update_all_documents_plot(
    fit_data: Dict, topic_names_data: Wrapped[List[str]]
) -> go.Figure:
    if (
        fit_data is None
        or topic_names_data is None
        or ("document_data" not in fit_data)
        or ("topic_names" not in topic_names_data)
    ):
        raise PreventUpdate()
    document_data = pd.DataFrame(fit_data["document_data"])
    names = pd.Series(topic_names_data["topic_names"])
    document_data = document_data.assign(
        topic_name=document_data.topic_id.map(names)
    )
    fig = documents_plot(document_data)
    return fig
