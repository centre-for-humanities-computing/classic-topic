"""Module containing all the callbacks of the application"""
import base64
import json
import sys
from typing import Callable, Dict, List, Tuple

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, ServersideOutput, State

from app.components.genre_weight_popup import (genre_weight_element,
                                               popup_container_class,
                                               setting_container_class)
from app.components.navbar import navbar_button_class
from app.components.settings import settings_hidden, settings_visible
from app.components.sidebar import sidebar_body_class, sidebar_shade_class
from app.components.toolbar import toolbar_class
from app.layout import view_class
from app.utils.metadata import fetch_metadata
from app.utils.modelling import (calculate_genre_importance,
                                 calculate_top_words, fit_pipeline,
                                 load_corpus, prepare_document_data,
                                 prepare_pipeline_data, prepare_topic_data,
                                 prepare_transformed_data, serialize_save_data)
from app.utils.plots import (all_topics_plot, document_topic_plot,
                             documents_plot, topic_plot)

# Global callback list
callbacks = []

corpus = load_corpus()


def cb(*args, **kwargs) -> Callable:
    """Decorator to add a function to the global callback list"""

    def _cb(func: Callable):
        callbacks.append({"function": func, "args": args, "kwargs": kwargs})
        return func

    return _cb


def add_callbacks(app: dash.Dash) -> None:
    """Adds the list of callbacks to a Dash app."""
    for callback in callbacks:
        app.callback(*callback["args"], **callback["kwargs"])(
            callback["function"]
        )


@cb(
    ServersideOutput("fit_store", "data"),
    Output("loading", "children"),
    Input("fit_pipeline", "n_clicks"),
    State("select_vectorizer", "value"),
    State("min_df", "value"),
    State("max_df", "value"),
    State("select_model", "value"),
    State("n_topics", "value"),
    Input("upload", "contents"),
    State("genre_weights", "data"),
    State("n_gram_slider", "value"),
    State("metadata", "data"),
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
    n_gram_range: List[int],
    metadata_store: Dict,
) -> Tuple[Dict, List]:
    """Updates fit data in the local store when the fit model button
    is pressed.
    """
    # print(ctx.triggered_id)
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
    if not n_clicks or not genre_weights:
        raise PreventUpdate
    # Fitting the topic pipeline
    metadata = pd.DataFrame.from_dict(metadata_store)
    n_gram_low, n_gram_high, *_ = n_gram_range
    pipeline = fit_pipeline(
        metadata=metadata,
        corpus=corpus,
        vectorizer_name=vectorizer_name,
        min_df=min_df,
        max_df=max_df,
        model_name=model_name,
        n_topics=n_topics,
        genre_weights=genre_weights,
        n_gram_range=(n_gram_low, n_gram_high),
    )
    # Inferring data from the fit
    genre_importance = calculate_genre_importance(corpus, pipeline)
    pipeline_data = prepare_pipeline_data(
        pipeline.vectorizer, pipeline.topic_model
    )
    transformed_data = prepare_transformed_data(
        pipeline.vectorizer, pipeline.topic_model, texts=corpus.text
    )
    topic_data = prepare_topic_data(**transformed_data, **pipeline_data)
    document_data = prepare_document_data(corpus=corpus, **transformed_data)
    pipeline_data.pop("components")
    fit_data = {
        "genre_importance": genre_importance.to_dict(),
        **pipeline_data,
        **topic_data,
        **document_data,
        "loaded": False,
    }
    # sizes = {
    #     key: sys.getsizeof(json.dumps(value))
    #     for key, value in fit_data.items()
    # }
    # print(sizes)
    return (
        fit_data,
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
def update_topic_switcher(topic_names: List[str], current_topic: int):
    """Updates the topic switcher component when the current topic changes.

    Parameters
    ----------
    topic_names: list of str
        Store data about topic names.
    current_topic: int
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
    if not topic_names:
        raise PreventUpdate
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
    Input("upload", "contents"),
    prevent_initial_call=True,
)
def update_topic_names(
    topic_names: List[str],
    current_topic: int,
    topic_name: str,
    fit_store: Dict,
    upload_contents: List,
) -> List[str]:
    """
    Updates topic names when the current topic name is changed or when a new
    model is fitted.

    Parameters
    ----------
    topic_names: list of str
        Store data of topic names.
    current_topic: int
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
        return [f"Topic {i}" for i in range(fit_store["n_topics"])]
    # If there is no topic names data prevent updating.
    if not topic_names:
        raise PreventUpdate()
    # Creating a list of new names
    new_names = topic_names.copy()
    new_names[current_topic] = topic_name
    return new_names


@cb(
    Output("sidebar_body", "className"),
    Output("topic_toolbar", "className"),
    Output("sidebar_shade", "className"),
    Input("sidebar_collapser", "n_clicks"),
    Input("current_view", "data"),
    prevent_initial_call=True,
)
def open_close_sidebar(
    n_clicks: int, current_view: str
) -> Tuple[str, str, str]:
    """Opens or closes sidebar and moves and hides the topic switcher.

    Parameters
    ----------
    n_clicks: int
        Number of times the sidebar collapse button has been clicked
    current_view: int
        Data about the current view.
    """
    hide_switcher = (
        " translate-y-0" if current_view == "topic" else " translate-y-full"
    )
    is_open = (n_clicks % 2) == 0

    if is_open:
        return (
            sidebar_body_class + " translate-x-full",
            toolbar_class + hide_switcher,
            sidebar_shade_class + " bg-opacity-0 hidden",
        )
    else:
        return (
            sidebar_body_class + " translate-x-0",
            toolbar_class + hide_switcher,
            sidebar_shade_class + " bg-opacity-40 block",
        )


@cb(
    Output("sidebar_collapser", "n_clicks"),
    State("sidebar_collapser", "n_clicks"),
    Input("fit_store", "data"),
    Input("fit_pipeline", "n_clicks"),
    Input("cancel_pipeline", "n_clicks"),
    prevent_initial_call=True,
)
def open_close_sidebar_fitting(
    current: int, fit_data: Dict, fit_clicks: int, cancel_clicks: int
) -> int:
    """Opens or closes sidebar when the fit pipeline button is pressed."""
    if (
        ((ctx.triggered_id == "fit_store") and (fit_data is None))
        or ((ctx.triggered_id == "fit_pipeline") and fit_clicks)
        or ((ctx.triggered_id == "cancel_pipeline") and cancel_clicks)
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
def update_current_view(topic_clicks: int, document_clicks: int) -> str:
    """Updates the current view value in store when a view is selected
    on the navbar."""
    if not topic_clicks and not document_clicks:
        raise PreventUpdate()
    if ctx.triggered_id == "topic_view_button":
        return "topic"
    if ctx.triggered_id == "document_view_button":
        return "document"
    raise PreventUpdate()


@cb(
    Output("topic_view_button", "className"),
    Output("document_view_button", "className"),
    Output("topic_view", "className"),
    Output("document_view", "className"),
    Input("current_view", "data"),
    prevent_initial_call=True,
)
def switch_views(current_view: str) -> Tuple[str, str, str, str]:
    """Switches views when the current view value in the store is changed."""
    if current_view == "topic":
        return (
            navbar_button_class + " text-sky-700",
            navbar_button_class + " text-gray-500",
            view_class + " flex mb-16",
            view_class + " hidden",
        )
    if current_view == "document":
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


@cb(
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


# @cb(
#     Output("current_topic_plot", "figure"),
#     Input("current_topic", "data"),
#     Input("fit_store", "data"),
#     prevent_initial_call=True,
# )
# def update_current_topic_plot(
#     current_topic: int, fit_store: Dict
# ) -> go.Figure:
#     """Updates the plots about the current topic in the topic view
#     when the current topic is changed or when a new model is fitted.
#     """
#     if current_topic is None or fit_store is None:
#         raise PreventUpdate()
#     genre_importance = pd.DataFrame(fit_store["genre_importance"])
#     top_words = pd.DataFrame(fit_store["top_words"])
#     return topic_plot(
#         current_topic, genre_importance=genre_importance, top_words=top_words
#     )


@cb(
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
    fig = all_topics_plot(topic_data, current_topic)
    return fig


@cb(
    Output("all_documents_plot", "figure"),
    Input("fit_store", "data"),
    Input("topic_names", "data"),
    Input("document_selector", "value"),
    State("all_documents_plot", "figure"),
)
def update_all_documents_plot(
    fit_data: Dict,
    topic_names: List[str],
    selected_id: int,
    current_fig: go.Figure,
) -> go.Figure:
    """Updates the document overview plot when a new model is fitted or when
    topic names are changed"""
    if not topic_names or fit_data is None:
        # If there's missing data, prevent update.
        raise PreventUpdate()
    document_data = pd.DataFrame(fit_data["document_data"])
    # Mapping topic names over to topic ids with a Series
    # since Series also function as a mapping, you can use them in the .map()
    # method
    names = pd.Series(topic_names)
    document_data = document_data.assign(
        topic_name=document_data.topic_id.map(names)
    )
    if ctx.triggered_id in ("fit_store", "topic_names"):
        return documents_plot(document_data)
    if ctx.triggered_id == "document_selector":
        if selected_id is None:
            raise PreventUpdate()
        selected_id = int(selected_id)
        selected_document = document_data[
            document_data.id_nummer == selected_id
        ].iloc[0]
        doc_name = f"{selected_document.værk} - {selected_document.forfatter}"
        layout = {
            **current_fig["layout"],
            "uirevision": True,
            "scene": {
                **current_fig["layout"]["scene"],
                "annotations": [
                    dict(
                        x=selected_document.x,
                        y=selected_document.y,
                        z=selected_document.z,
                        text=doc_name,
                        bgcolor="white",
                        bordercolor="black",
                        arrowsize=1,
                        arrowwidth=2,
                        borderwidth=3,
                        borderpad=10,
                        font=dict(size=16, color="#0369a1"),
                    )
                ],
            },
        }
        data = current_fig["data"]
        fig = go.Figure(data=data, layout=layout)
        return fig
    raise PreventUpdate()


@cb(
    Output("download", "data"),
    Input("download_button", "n_clicks"),
    State("fit_store", "data"),
    State("topic_names", "data"),
    prevent_initial_call=True,
)
def download_data(
    n_clicks: int, fit_data: Dict, topic_names: List[str]
) -> Dict:
    if not n_clicks:
        raise PreventUpdate()
    if not fit_data or not topic_names:
        raise PreventUpdate()
    return {
        "content": serialize_save_data(fit_data, topic_names),
        "filename": "model_data.json",
    }


@cb(
    Output("metadata", "data"),
    Input("fetch_data", "n_intervals"),
)
def fetch_data(n_intervals: int) -> Dict:
    """Fetches metadata and puts it into a dash store."""
    # print("Fetching metadata")
    metadata = fetch_metadata()
    metadata = metadata.assign(group=metadata.group.fillna("Rest"))
    return metadata.to_dict()


@cb(
    Output("genre_weight_popup", "children"),
    Input("metadata", "data"),
    prevent_initial_call=True,
)
def update_genre_weights_popup_children(metadata_store: Dict) -> List:
    """Updates the children of the genre weights popup"""
    metadata: pd.DataFrame = pd.DataFrame.from_dict(metadata_store)
    genres = metadata.group.unique()
    return [genre_weight_element(genre) for genre in genres]


@cb(
    Output("genre_weights", "data"),
    Input(dict(type="genre_switch", index=dash.ALL), "on"),
    State(dict(type="genre_switch", index=dash.ALL), "id"),
    Input(dict(type="genre_weight_slider", index=dash.ALL), "value"),
    State(dict(type="genre_weight_slider", index=dash.ALL), "id"),
    prevent_initial_call=True,
)
def update_genre_weights(
    is_on: List[bool],
    switch_ids: List[Dict[str, str]],
    weights: List[int],
    weight_ids: List[Dict[str, str]],
):
    """Updates genre weights based on the values set in the popup."""
    if not is_on or not switch_ids or not weights or not weight_ids:
        raise PreventUpdate()
    is_on_mapping = {id["index"]: on for id, on in zip(switch_ids, is_on)}
    # print("is on mapping:", is_on_mapping)
    weight_mapping = {
        id["index"]: weight for id, weight in zip(weight_ids, weights)
    }
    # print("weight mapping:", weight_mapping)
    genres = is_on_mapping.keys()
    genre_weights = {
        genre: weight_mapping[genre] if is_on_mapping[genre] else 0
        for genre in genres
    }
    # print("Changing genre weights:")
    # print(genre_weights)
    return genre_weights


@cb(
    Output("document_selector", "options"),
    Input("metadata", "data"),
)
def update_document_selector_options(metadata_store: Dict) -> Dict[int, str]:
    metadata = pd.DataFrame.from_dict(metadata_store)
    metadata = metadata.merge(corpus, on="id_nummer", how="inner")
    return {
        int(id_nummer): f"{work} - {author}"
        for id_nummer, work, author in zip(
            metadata.id_nummer, metadata.værk, metadata.forfatter
        )
    }


@cb(
    Output("document_selector", "value"),
    Input("all_documents_plot", "clickData"),
)
def select_document(selected_points: Dict) -> int:
    if not selected_points:
        raise PreventUpdate()
    point, *_ = selected_points["points"]
    text_id = point["customdata"][-1]
    return int(text_id)


@cb(
    Output("document_genre", "children"),
    Output("document_group", "children"),
    Output("document_topics_graph", "figure"),
    Output("document_content", "children"),
    Input("document_selector", "value"),
    State("metadata", "data"),
    State("fit_store", "data"),
    State("topic_names", "data"),
)
def update_document_inspector(
    id_nummer: int,
    metadata_store: Dict,
    fit_data: Dict,
    topic_names: List[str],
) -> Tuple[str, str, go.Figure, str]:
    if id_nummer is None:
        raise PreventUpdate()
    id_nummer = int(id_nummer)
    document_data = (
        pd.DataFrame(fit_data["document_data"])
        .set_index("id_nummer")
        .loc[id_nummer]
    )
    i_doc = document_data.i_doc
    importances = pd.DataFrame(fit_data["document_topic_importance"])
    print(importances)
    importances = importances[importances.i_doc == i_doc]
    fig = document_topic_plot(importances, topic_names)
    genre = f"Genre: {document_data.tlg_genre}"
    group = f"Group: {document_data.group}"
    return (genre, group, fig, document_data.text)


@cb(
    Output("genre_weight_popup_container", "className"),
    Input("weight_settings", "n_clicks"),
    Input("close_weight_popup", "n_clicks"),
    prevent_initial_call=True,
)
def open_close_genre_weights_popup(open_clicks: int, close_clicks: int) -> str:
    """Opens and closes genre weights popup when needed"""
    if not open_clicks and not close_clicks:
        raise PreventUpdate()
    if "weight_settings" == ctx.triggered_id:
        return popup_container_class + " fixed"
    if "close_weight_popup" == ctx.triggered_id:
        return popup_container_class + " hidden"
    raise PreventUpdate()


@cb(
    Output(
        dict(type="genre_settings_container", index=dash.MATCH), "className"
    ),
    Input(dict(type="genre_switch", index=dash.MATCH), "on"),
    prevent_initial_call=True,
)
def hide_genre_settings(is_genre_on: bool) -> str:
    """Hides settings for a genre if it is filtered away in the popup"""
    if is_genre_on:
        return setting_container_class
    else:
        return setting_container_class + " hidden"


@cb(
    Output(
        dict(type="_setting_group_body", index=dash.MATCH),
        "className",
    ),
    Output(
        dict(type="_setting_group_collapse", index=dash.MATCH),
        "className",
    ),
    Input(dict(type="_setting_group_collapse", index=dash.MATCH), "n_clicks"),
    prevent_initial_call=True,
)
def expand_hide_setting_group(n_clicks: int) -> Tuple[str, str]:
    is_on = not (n_clicks % 2)
    if is_on:
        return settings_visible, "transition-all ease-in rotate-0"
    else:
        return "hidden", "transition-all ease-in rotate-180"
