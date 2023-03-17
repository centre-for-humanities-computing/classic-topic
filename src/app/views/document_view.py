from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from dash import ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State, dcc, html

from app.components import document_inspector
from app.utils.callback import init_callbacks
from app.utils.plots import documents_plot

callbacks, def_callback = init_callbacks()
callbacks.extend(document_inspector.callbacks)

view_class = "flex-row items-stretch flex-1 mr-16 z-0"

layout = html.Div(
    id="document_view",
    className=view_class + " hidden",
    children=[
        document_inspector.layout,
        dcc.Graph(
            id="all_documents_plot",
            className="flex-1 basis-2/3",
            responsive=True,
            animate=True,
            animation_options=dict(frame=dict(redraw=True)),
        ),
    ],
)


@def_callback(
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
        selected_document = document_data[
            document_data.document_id == selected_id
        ].iloc[0]
        doc_name = f"{selected_document.work} - {selected_document.author}"
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
                        bgcolor="rgba(255,255,255,0.75)",
                        arrowsize=1,
                        arrowwidth=2,
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
