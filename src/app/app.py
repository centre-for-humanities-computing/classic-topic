"""Describes the app layout and callbacks"""
import base64
import json
from numbers import Integral
from typing import Dict, List, Tuple

import pandas as pd
from dash import ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import (Input, Output, ServersideOutput, State,
                                    dcc, html)

from app.components import (accordion, genre_weight_popup, navbar, sidebar,
                            toolbar)
from app.utils.callback import init_callbacks
from app.utils.modelling import (calculate_genre_importance, fit_pipeline,
                                 prepare_corpus, prepare_document_data,
                                 prepare_pipeline_data, prepare_topic_data,
                                 prepare_transformed_data)
from app.views import document_view, topic_view

# Loading corpus from disk
corpus = pd.read_csv("../dat/cleaned_corpus.csv")

callbacks, def_callback = init_callbacks()
callbacks.extend(topic_view.callbacks)
callbacks.extend(document_view.callbacks)
callbacks.extend(toolbar.callbacks)
callbacks.extend(sidebar.callbacks)
callbacks.extend(navbar.callbacks)
callbacks.extend(genre_weight_popup.callbacks)
callbacks.extend(accordion.callbacks)


layout = html.Div(
    className="flex flex-row w-full h-full fixed",
    children=[
        dcc.Store(id="fit_store", storage_type="session"),
        dcc.Store(id="topic_names", data=[], storage_type="session"),
        dcc.Store(id="genre_weights", storage_type="session"),
        dcc.Store(
            id="current_view",
            storage_type="session",
            data="topic",
        ),
        dcc.Loading(
            type="circle",
            className="flex flex-1 mr-16 z-0",
            fullscreen=True,
            children=html.Div(id="loading"),
        ),
        topic_view.layout,
        document_view.layout,
        toolbar.layout,
        sidebar.layout,
        navbar.layout,
        genre_weight_popup.layout,
    ],
)


@def_callback(
    Output("fit_pipeline", "className"),
    Output("fit_pipeline", "disabled"),
    Output("fit_pipeline", "title"),
    Input("min_df", "value"),
    Input("max_df", "value"),
    Input("genre_weights", "data"),
)
def enable_disable_fit_button(
    min_df: int, max_df: float, genre_weights: Dict[str, int]
) -> Tuple[str, bool, str]:
    """Disables button when max_df and min_df are inappropriate"""
    enabled_style = """
        basis-2/3
        text-white bg-green-600
        hover:bg-green-700
        h-12 flex-1 text-center 
        transition-all ease-in
        rounded-xl shadow
    """
    disabled_style = """
        basis-2/3
        text-black bg-gray-600
        h-12 flex-1 text-center 
        transition-all ease-in
        rounded-xl shadow
    """
    max_df = float(max_df)
    weighted_corpus = prepare_corpus(corpus, genre_weights)
    n_doc = len(weighted_corpus.index)
    max_doc_count = max_df if isinstance(max_df, Integral) else max_df * n_doc
    min_doc_count = min_df if isinstance(min_df, Integral) else min_df * n_doc
    if max_doc_count < min_doc_count:
        alt_message = f"""
        Maximum number of documents({max_doc_count})
        is smaller Than minimum number of documents({min_doc_count}).
        Either increase max_df or lower min_df.
        """
        return disabled_style, True, alt_message
    else:
        return enabled_style, False, "Fit topic model"


@def_callback(
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
) -> Tuple[Dict, List]:
    """Updates fit data in the local store when the fit model button
    is pressed.
    """
    max_df = float(max_df)
    print("Updating fit")
    print(f"\t{n_clicks=}")
    print(f"\t{vectorizer_name=}")
    print(f"\t{min_df=}")
    print(f"\t{max_df=}")
    print(f"\t{model_name=}")
    print(f"\t{n_topics=}")
    print(f"\t{genre_weights=}")
    print(f"\t{n_gram_range=}")
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
    weighted_corpus = prepare_corpus(corpus, genre_weights)
    # Fitting the topic pipeline
    n_gram_low, n_gram_high, *_ = n_gram_range
    pipeline = fit_pipeline(
        corpus=weighted_corpus,
        vectorizer_name=vectorizer_name,
        min_df=min_df,
        max_df=max_df,
        model_name=model_name,
        n_topics=n_topics,
        n_gram_range=(n_gram_low, n_gram_high),
    )
    # Inferring data from the fit
    genre_importance = calculate_genre_importance(weighted_corpus, pipeline)
    vectorizer = pipeline.named_steps["vectorizer"]
    topic_model = pipeline.named_steps["topic_model"]
    pipeline_data = prepare_pipeline_data(vectorizer, topic_model)
    transformed_data = prepare_transformed_data(
        vectorizer, topic_model, texts=weighted_corpus.text
    )
    topic_data = prepare_topic_data(**transformed_data, **pipeline_data)
    document_data = prepare_document_data(
        corpus=weighted_corpus, **transformed_data
    )
    pipeline_data.pop("components")
    fit_data = {
        "genre_importance": genre_importance.to_dict(),
        **pipeline_data,
        **topic_data,
        **document_data,
        "loaded": False,
    }
    return (
        fit_data,
        [],
    )


@def_callback(
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
    """
    print("Updating topic names")
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


@def_callback(
    Output("document_selector", "options"),
    Input("fit_store", "data"),
)
def update_document_selector_options(fit_data: Dict) -> Dict[int, str]:
    # This callback should conceptually belong to document inspector
    # it has to be here since the corpus is only loaded in this module.
    # Consider putting the corpus in a server Store and moving this to
    # document inspector.
    print("Updating doc inspector options")
    if fit_data is None:
        raise PreventUpdate
    documents = pd.DataFrame.from_dict(fit_data["document_data"])
    documents = documents.merge(corpus, on="document_id", how="inner")
    return {
        document_id: f"{work} - {author}"
        for document_id, work, author in zip(
            documents.document_id, documents.work, documents.author
        )
    }
