"""Module containing all the callbacks of the application"""
import base64
import json
from typing import Callable, Dict, List, Tuple, TypeVar

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.components.sidebar import sidebar_body_class
from app.components.topic_switcher import topic_switcher_class
from app.components.navbar import navbar_button_class
from app.components.genre_weight_popup import popup_container_class
from app.layout import view_class
from app.utils.modelling import (
    calculate_document_data,
    calculate_genre_importance,
    calculate_top_words,
    calculate_topic_data,
    fit_pipeline,
    load_corpus,
    serialize_save_data,
)
from app.utils.plots import all_topics_plot, documents_plot, topic_plot
from app.utils.metadata import fetch_metadata

# Global callback list
callbacks = []

T = TypeVar("T")

Wrapped = Dict[str, T]


def cb(*args, **kwargs) -> Callable:
    """Decorator to add a function to the global callback list"""

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
    Input("upload", "contents"),
    State("genre_weights", "data"),
    prevent_initial_call=True,
)
def update_fit(
    n_clicks: int,
    vectorizer_name: str,
    min_df: int,
    max_df: float,
    model_name: str,
    n_topics: int,
    upload_contents: List,
    genre_weights: Dict[str, int],
) -> Tuple[Dict, List]:
    """Updates fit data in the local store when the fit model button is pressed.

    Parameters
    ----------
    n_clicks: int
        Number of times the 'fit_pipeline' button has been pressed.
    vectorizer_name: {'tf-idf', 'bow'}
        Describes whether a TF-IDF of Bag of Words vectorizer should be fitted.
    min_df: int
        Minimum document frequency parameter of the vectorizer.
    max_df: float
        Minimum document frequency parameter of the vectorizer.
    model_name: {'nmf', 'lda', 'lsa'/'lsi', 'dmm'}
        Specifies which topic model should be trained on the corpus.
    n_topics: int
        Number of topics the model should find.
    genre_weights: dict of str to int
        Weights of the different genres.

    Returns
    -------
    fit_store.data: dict
        Data about the model fit.
    loading.children: list
        Empty list, is returned so that the loading component activates
        while the callback is executed.
    """
    if ctx.triggered_id == "upload":
        if not upload_contents:
            raise PreventUpdate()
        # If the data has been uploaded, parse contents and return them
        _, content_string = upload_contents.split(",")
        # Decoding data
        decoded = base64.b64decode(content_string)
        text = decoded.decode("utf-8")
        data = json.loads(text)
        return data["fit_data"], []
    # If the button has not actually been clicked prevent updating
    if not n_clicks:
        raise PreventUpdate
    # Load the corpus from disk
    corpus = load_corpus()
    # Fitting the topic pipeline
    pipeline = fit_pipeline(
        corpus=corpus,
        vectorizer_name=vectorizer_name,
        min_df=min_df,
        max_df=max_df,
        model_name=model_name,
        n_topics=n_topics,
        genre_weights=genre_weights,
    )
    # Inferring data from the fit
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
            "loaded": False,
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
    """Updates the topic switcher component when the current topic changes.

    Parameters
    ----------
    topic_names_data: wrapped list of str
        Store data about topic names.
    current_topic_data: wrapped int
        Store data about currently selected topic.

    Returns
    -------
    next_topic.children
        Text that should be displayed on the next topic button.
    next_topic.disabled
        Whether the next topic button should be disabled or not.
    prev_topic.children
        Text that should be displayed on the previous topic button.
    prev_topic.disabled
        Whether the previous topic button should be disabled or not.
    topic_name.value
        List of topic names.
    """
    if topic_names_data is None or current_topic_data is None:
        raise PreventUpdate
    topic_names = topic_names_data["topic_names"]
    current_topic = current_topic_data["current_topic"]
    n_topics = len(topic_names)
    current = topic_names[current_topic]
    prev_disabled = current_topic == 0
    next_disabled = current_topic == n_topics - 1
    prev_topic = "" if prev_disabled else "<- " + topic_names[current_topic - 1]
    next_topic = "" if next_disabled else topic_names[current_topic + 1] + " ->"
    return next_topic, next_disabled, prev_topic, prev_disabled, current


@cb(
    Output("topic_names", "data"),
    State("topic_names", "data"),
    State("current_topic", "data"),
    Input("topic_name", "value"),
    Input("fit_store", "data"),
    Input("upload", "contents"),
    prevent_initial_call=True,
)
def update_topic_names(
    topic_names_data: Wrapped[List[str]],
    current_topic_data: Wrapped[int],
    topic_name: str,
    fit_store: Dict,
    upload_contents: List,
) -> Wrapped[List[str]]:
    """
    Updates topic names when the current topic name is changed or when a new model is fitted.

    Parameters
    ----------
    topic_names_data: wrapped list of str
        Store data of topic names.
    current_topic_data: wrapped int
        Store data of current topic.
    topic_name: str
        Current value of the topic name entry component.
    fit_store: dict
        Data about the model fit.

    Returns
    -------
    topic_names.data
        List of topic names.
    """
    if ctx.triggered_id == "upload":
        if not upload_contents:
            raise PreventUpdate()
        # If the data has been uploaded, parse contents and return them
        _, content_string = upload_contents.split(",")
        # Decoding data
        decoded = base64.b64decode(content_string)
        text = decoded.decode("utf-8")
        data = json.loads(text)
        return data["topic_names"]
    # Check if the callback has been triggered by the fit updating.
    if ctx.triggered_id == "fit_store":
        # If the store is empty prevent updating.
        if fit_store is None or fit_store["loaded"]:
            raise PreventUpdate()
        # Return a list of default topic names.
        return {"topic_names": [f"Topic {i}" for i in range(fit_store["n_topics"])]}
    # If there is no topic names data prevent updating.
    if topic_names_data is None or current_topic_data is None:
        raise PreventUpdate()
    # Unwrapping data
    topic_names = topic_names_data["topic_names"]
    current_topic = current_topic_data["current_topic"]
    if not topic_names:
        raise PreventUpdate()
    # Creating a list of new names
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
    """Opens or closes sidebar and moves and hides the topic switcher.

    Parameters
    ----------
    n_clicks: int
        Number of times the sidebar collapse button has been clicked
    current_view_data: wrapped int
        Data about the current view.

    Returns
    -------
    sidebar_body.className: str
        Indicates style and position of the sidebar body.
    topic_switcher.className: str
        Describes position and style for the topic switcher.
    sidebar_collapser.children: str
        The sidebar collapser button icon.
    """
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
def open_close_sidebar_fitting(current: int, fit_data: Dict, n_clicks: int) -> int:
    """Opens or closes sidebar when the fit pipeline button is pressed."""
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
def update_current_view(topic_clicks: int, document_clicks: int) -> Wrapped[str]:
    """Updates the current view value in store when a view is selected on the navbar."""
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
    """Switches views when the current view value in the store is changed."""
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
    """Updates current topic in the store when one is selected."""
    if "fit_store" == ctx.triggered_id:
        return {"current_topic": 0}
    if current_topic_data is None or "current_topic" not in current_topic_data:
        raise PreventUpdate()
    current_topic = current_topic_data["current_topic"]
    if "all_topics_plot" == ctx.triggered_id:
        if plot_click_data is None:
            raise PreventUpdate()
        # In theory multiple points could be selected with
        # multiple customdata elements, so we unpack the first element.
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
    """Updates the plots about the current topic in the topic view
    when the current topic is changed or when a new model is fitted.
    """
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
    """Updates the topic overview plot when the fit, the topic names or the current topic change."""
    if (
        fit_data is None
        or topic_names_data is None
        or current_topic_data is None
        or ("topic_data" not in fit_data)
        or ("topic_names" not in topic_names_data)
        or ("current_topic" not in current_topic_data)
    ):
        # If there's missing data, prevent update.
        raise PreventUpdate()
    topic_data = pd.DataFrame(fit_data["topic_data"])
    current_topic = current_topic_data["current_topic"]
    # Mapping topic names over to topic ids with a Series
    # since Series also function as a mapping, you can use them in the .map() method
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
    """Updates the document overview plot when a new model is fitted or when topic names are changed"""
    if (
        fit_data is None
        or topic_names_data is None
        or ("document_data" not in fit_data)
        or ("topic_names" not in topic_names_data)
    ):
        # If there's missing data, prevent update.
        raise PreventUpdate()
    document_data = pd.DataFrame(fit_data["document_data"])
    # Mapping topic names over to topic ids with a Series
    # since Series also function as a mapping, you can use them in the .map() method
    names = pd.Series(topic_names_data["topic_names"])
    document_data = document_data.assign(topic_name=document_data.topic_id.map(names))
    fig = documents_plot(document_data)
    return fig


@cb(
    Output("download", "data"),
    Input("download_button", "n_clicks"),
    State("fit_store", "data"),
    State("topic_names", "data"),
    prevent_initial_call=True,
)
def download_data(n_clicks: int, fit_data: Dict, topic_names: Wrapped[str]) -> Dict:
    if not n_clicks:
        raise PreventUpdate()
    if not fit_data or not topic_names:
        raise PreventUpdate()
    return {
        "content": serialize_save_data(fit_data, topic_names),
        "filename": "model_data.json",
    }


@cb(
    Output("genre_names", "data"),
    Output("fetch_genres", "disabled"),
    Input("fetch_genres", "n_intervals"),
)
def update_genre_names(n_intervals: int) -> Tuple[List[str], bool]:
    """Fetches genre names from the metadata chart, and
    disables updating afterwards"""
    metadata = fetch_metadata()
    metadata = metadata.assign(group=metadata.group.fillna("Rest"))
    genres = metadata.group.unique()
    # print(f"Fetched genres: {genres}")
    return genres.tolist(), True

@cb(
    Output("genre_weights_slider", "value"),
    Input("genre_weights_dropdown", "value"),
    State("genre_weights", "data"),
    prevent_initial_call=True,
)
def update_genre_weights_slider_value(selected: str, weights: Dict[str, int]) -> int:
    """Updates genre weight slider value when another genre is selected
    """
    if not selected or not weights:
        raise PreventUpdate()
    return weights[selected]

@cb(
    Output("genre_weights_dropdown", "value"),
    Output("genre_weights_dropdown", "options"),
    Input("genre_names", "data"),
    prevent_initial_call=True,
)
def set_genre_weight_dropdown_options(genre_names: List[str]) -> Tuple[str, List[str]]:
    """Sets genre weight dropdown options and current value,
    once genre names are loaded"""
    if not genre_names:
        raise PreventUpdate()
    return genre_names[0], genre_names

@cb(
    Output("genre_weights", "data"),
    Input("genre_names", "data"),
    Input("genre_weights_slider", "value"),
    State("genre_weights_dropdown", "value"),
    State("genre_weights", "data"),
    prevent_initial_call=True,
)
def update_genre_weights(
    genre_names: List[str],
    update_value: int,
    selected: str,
    prev_weights: Dict[str, int],
) -> Tuple[Dict[str, int], bool]:
    """Updates genre weights."""
    # print("Updating genre weights")
    if not genre_names:
        raise PreventUpdate()
    if ctx.triggered_id == "genre_names":
        genre_weights = {genre_name: 1 for genre_name in genre_names}
        return genre_weights
    if ctx.triggered_id == "genre_weights_slider":
        if not selected:
            raise PreventUpdate()
        genre_weights = {**prev_weights, selected: update_value}
        return genre_weights
    raise PreventUpdate()


# @cb(
#     Output("genre_weight_popup_container", "className"),
#     Input("weight_settings", "n_clicks"),
#     Input("close_weight_popup", "n_clicks"),
#     prevent_initial_call=True,
# )
# def open_close_genre_weights_popup(open_clicks: int, close_clicks: int) -> str:
#     """Opens and closes genre weights popup when needed"""
#     if not open_clicks and not close_clicks:
#         raise PreventUpdate()
#     if "weight_settings" == ctx.triggered_id:
#         return popup_container_class + " fixed"
#     if "close_weight_popup" == ctx.triggered_id:
#         return popup_container_class + " hidden"
#     raise PreventUpdate()
