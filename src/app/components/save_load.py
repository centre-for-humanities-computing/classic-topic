"""Component for saving and loading fit data."""
from typing import List, Dict

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import html, dcc
from dash_extensions.enrich import Input, Output, State

from app.utils.callback import init_callbacks
from app.utils.modelling import serialize_save_data

callbacks, def_callback = init_callbacks()


button_class = """
    text-xl transition-all ease-in 
    justify-center content-center items-center
    text-gray-500 hover:text-sky-600
    flex flex-1
"""

layout = html.Div(
    className="""
        flex flex-none flex-row justify-center content-middle
        h-16 w-32 bg-white shadow rounded-full
        rounded-full
    """,
    children=[
        html.Button(
            html.I(className="fa-solid fa-file-arrow-down"),
            id="download_button",
            title="Download data",
            className=button_class,
        ),
        dcc.Download(id="download"),
        html.Div(
            className=button_class,
            children=dcc.Upload(
                id="upload",
                children=html.Button(
                    html.I(className="fa-solid fa-file-arrow-up"),
                    id="upload_button",
                    title="Upload data",
                    # className=button_class,
                ),
            ),
        ),
    ],
)


@def_callback(
    Output("download", "data"),
    Input("download_button", "n_clicks"),
    State("fit_store", "data"),
    State("topic_names", "data"),
    prevent_initial_call=True,
)
def download_data(n_clicks: int, fit_data: Dict, topic_names: List[str]) -> Dict:
    if not n_clicks:
        raise PreventUpdate()
    if not fit_data or not topic_names:
        raise PreventUpdate()
    return {
        "content": serialize_save_data(fit_data, topic_names),
        "filename": "model_data.json",
    }
