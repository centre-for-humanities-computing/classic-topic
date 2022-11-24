"""Module describing the component for inspecting documents"""
from typing import Dict, List, Tuple

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import dcc, html
from dash_extensions.enrich import Input, Output, ServersideOutput, State
import pandas as pd
import plotly.graph_objects as go

from app.components import accordion
from app.utils.callback import init_callbacks
from app.utils.plots import document_topic_plot

callbacks, def_callback = init_callbacks()

layout = html.Div(
    className="""basis-1/3 flex-1 flex-col bg-white shadow
    overflow-y-scroll overflow-x-hidden p-5 space-y-5
    """,
    children=[
        dcc.Dropdown(
            id="document_selector",
            options={},
            value=None,
        ),
        accordion.Accordion(
            name="Information",
            index="inspector_info",
            children=[
                html.Ul(
                    [
                        html.Li(
                            id="document_genre",
                            children="TLG genre: None",
                            className="text-lg py-2",
                        ),
                        html.Li(
                            id="document_group",
                            children="Ground: None",
                            className="text-lg py-2",
                        ),
                    ],
                    className="list-disc pl-8",
                ),
            ],
        ),
        accordion.Accordion(
            "Topics",
            index="inspector_topics",
            children=[
                dcc.Graph(id="document_topics_graph", animate=False),
            ],
        ),
        accordion.Accordion(
            "Content",
            index="inspector_content",
            children=[
                html.Div(
                    id="document_content",
                    children="This is the textual content of the document",
                    className="""
                text-justify h-1/3
                """,
                ),
            ],
        ),
    ],
)


@def_callback(
    Output("document_genre", "children"),
    Output("document_group", "children"),
    Output("document_topics_graph", "figure"),
    Output("document_content", "children"),
    Input("document_selector", "value"),
    State("fit_store", "data"),
    State("topic_names", "data"),
)
def update_document_inspector(
    id_nummer: int,
    fit_data: Dict,
    topic_names: List[str],
) -> Tuple[str, str, go.Figure, str]:
    """Updates plots and data in the document inspector when a document is
    selected"""
    if id_nummer is None:
        # Prevent updating if no document is selected
        raise PreventUpdate()
    # Making sure the ID is a number
    id_nummer = int(id_nummer)
    # Finding data for the selected document in the DataFrame
    document_data = (
        pd.DataFrame(fit_data["document_data"]).set_index("id_nummer").loc[id_nummer]
    )
    # Extracting index of the document
    i_doc = document_data.i_doc
    # Getting topic importances for the document
    importances = pd.DataFrame(fit_data["document_topic_importance"])
    importances = importances[importances.i_doc == i_doc]
    # Producing plot
    fig = document_topic_plot(importances, topic_names)
    genre = f"Genre: {document_data.tlg_genre}"
    group = f"Group: {document_data.group}"
    return (genre, group, fig, document_data.text)


@def_callback(
    Output("document_selector", "value"),
    Input("all_documents_plot", "clickData"),
)
def select_document(selected_points: Dict) -> int:
    """Selects document when it is clicked on in the scatter plot"""
    if not selected_points:
        raise PreventUpdate()
    point, *_ = selected_points["points"]
    text_id = point["customdata"][-1]
    return int(text_id)
